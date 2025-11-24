#!/usr/bin/env python3
"""
Omni3D数据读取脚本 - 独立版本（不依赖detectron2）

基于cubercnn/data/datasets.py的核心逻辑，提取出独立的数据读取功能。
只需要: pycocotools, numpy, json
"""

import json
import os
import time
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO


# ==================== BBox格式转换工具 ====================
class BoxMode:
    """BBox格式转换工具类（从detectron2提取的核心逻辑）"""
    XYXY_ABS = 0  # [x1, y1, x2, y2] 绝对坐标
    XYWH_ABS = 1  # [x, y, w, h] 绝对坐标，x,y是左上角
    
    @staticmethod
    def convert(box, from_mode, to_mode):
        """
        转换bbox格式
        Args:
            box: list或array，bbox坐标
            from_mode: 源格式 (BoxMode.XYXY_ABS 或 BoxMode.XYWH_ABS)
            to_mode: 目标格式
        Returns:
            converted_box: 转换后的bbox
        """
        box = np.asarray(box, dtype=np.float32)
        
        if from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
            # [x1, y1, x2, y2] -> [x, y, w, h]
            x1, y1, x2, y2 = box
            return np.array([x1, y1, x2 - x1, y2 - y1])
        
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYXY_ABS:
            # [x, y, w, h] -> [x1, y1, x2, y2]
            x, y, w, h = box
            return np.array([x, y, x + w, y + h])
        
        else:
            raise ValueError(f"Unsupported conversion: {from_mode} -> {to_mode}")


# ==================== 数据过滤设置 ====================
def get_default_filter_settings():
    """获取默认的过滤设置"""
    return {
        'category_names': [],           # 类别名称列表，空列表表示使用所有类别
        'ignore_names': [],              # 要忽略的类别名称
        'truncation_thres': 0.99,        # 截断阈值
        'visibility_thres': 0.01,       # 可见度阈值
        'min_height_thres': 0.00,        # 最小高度阈值（相对于图片高度）
        'max_height_thres': 1.50,        # 最大高度阈值（相对于图片高度）
        'modal_2D_boxes': False,         # 是否使用tight 2D boxes
        'trunc_2D_boxes': False,          # 是否使用truncated 2D boxes
        'max_depth': 100,                # 最大深度
        'min_depth': 0.2,                # 最小深度
    }


def is_ignore_annotation(anno, filter_settings, image_height):
    """
    判断标注是否应该被忽略
    基于cubercnn/data/datasets.py的is_ignore函数
    """
    ignore = anno.get('behind_camera', False)
    ignore |= (not bool(anno.get('valid3D', True)))
    
    if ignore:
        return ignore
    
    # 检查尺寸
    dimensions = anno.get('dimensions', [0, 0, 0])
    ignore |= dimensions[0] <= 0
    ignore |= dimensions[1] <= 0
    ignore |= dimensions[2] <= 0
    
    # 检查深度
    center_cam = anno.get('center_cam', [0, 0, 1e9])
    ignore |= center_cam[2] > filter_settings['max_depth']
    ignore |= center_cam[2] < filter_settings['min_depth']
    
    # 检查点云和分割点
    ignore |= (anno.get('lidar_pts', 0) == 0)
    ignore |= (anno.get('segmentation_pts', 0) == 0)
    ignore |= (anno.get('depth_error', 0) > 0.5)
    
    # 获取2D bbox
    bbox2D = None
    
    # 优先使用tight boxes
    if filter_settings['modal_2D_boxes'] and 'bbox2D_tight' in anno and anno['bbox2D_tight'][0] != -1:
        bbox2D = BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    
    # 其次使用truncated boxes
    elif filter_settings['trunc_2D_boxes'] and 'bbox2D_trunc' in anno and not np.all([val == -1 for val in anno['bbox2D_trunc']]):
        bbox2D = BoxMode.convert(anno['bbox2D_trunc'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    
    # 最后使用projected boxes
    elif 'bbox2D_proj' in anno:
        bbox2D = BoxMode.convert(anno['bbox2D_proj'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    
    if bbox2D is None:
        return True  # 没有有效的2D bbox，忽略
    
    # 检查高度
    height = bbox2D[3]
    ignore |= height <= filter_settings['min_height_thres'] * image_height
    ignore |= height >= filter_settings['max_height_thres'] * image_height
    
    # 检查截断和可见度
    truncation = anno.get('truncation', -1)
    visibility = anno.get('visibility', -1)
    
    if truncation >= 0:
        ignore |= truncation >= filter_settings['truncation_thres']
    
    if visibility >= 0:
        ignore |= visibility <= filter_settings['visibility_thres']
    
    # 检查忽略类别
    if 'ignore_names' in filter_settings:
        category_name = anno.get('category_name', '')
        ignore |= category_name in filter_settings['ignore_names']
    
    return ignore


# ==================== Omni3D数据加载类 ====================
class Omni3DLoader:
    """
    Omni3D数据加载器
    基于cubercnn/data/datasets.py的Omni3D类，但不依赖detectron2
    """
    
    def __init__(self, annotation_files, data_root=None, filter_settings=None):
        """
        初始化数据加载器
        Args:
            annotation_files: JSON文件路径（字符串或列表）
            data_root: 数据根目录，用于拼接图片路径
            filter_settings: 过滤设置字典，None表示使用默认设置
        """
        if isinstance(annotation_files, str):
            annotation_files = [annotation_files]
        
        self.data_root = data_root
        self.filter_settings = filter_settings or get_default_filter_settings()
        
        # 加载数据集
        self.dataset = {}
        self.images = []
        self.annotations = []
        self.categories = []
        self.imgToAnns = defaultdict(list)
        self.catToImgs = defaultdict(list)
        
        cats_ids_master = []
        cats_master = []
        
        for annotation_file in annotation_files:
            print(f'Loading {os.path.basename(annotation_file)} annotations...')
            tic = time.time()
            
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            
            assert isinstance(dataset, dict), f'Invalid annotation format: {type(dataset)}'
            print(f'Done (t={time.time() - tic:.2f}s)')
            
            # 处理info字段
            if isinstance(dataset['info'], list):
                dataset['info'] = dataset['info'][0]
            
            dataset['info']['known_category_ids'] = [cat['id'] for cat in dataset['categories']]
            
            # 合并数据集
            if len(self.dataset) == 0:
                self.dataset = dataset
            else:
                if isinstance(self.dataset['info'], dict):
                    self.dataset['info'] = [self.dataset['info']]
                self.dataset['info'] += [dataset['info']]
                self.dataset['annotations'] += dataset['annotations']
                self.dataset['images'] += dataset['images']
            
            # 收集类别
            for cat in dataset['categories']:
                if cat['id'] not in cats_ids_master:
                    cats_ids_master.append(cat['id'])
                    cats_master.append(cat)
        
        # 处理类别
        if self.filter_settings is None or len(self.filter_settings.get('category_names', [])) == 0:
            # 使用所有类别
            self.categories = [
                cats_master[i] 
                for i in np.argsort(cats_ids_master)
            ]
            if self.filter_settings:
                self.filter_settings['category_names'] = [cat['name'] for cat in self.categories]
        else:
            # 只使用指定的类别
            self.categories = [
                cats_master[i] 
                for i in np.argsort(cats_ids_master)
                if cats_master[i]['name'] in self.filter_settings['category_names']
            ]
        
        # 过滤标注
        if self.filter_settings:
            self._filter_annotations()
        
        # 创建索引
        self._create_index()
    
    def _filter_annotations(self):
        """过滤标注"""
        valid_anns = []
        im_height_map = {img['id']: img['height'] for img in self.dataset['images']}
        
        trainable_cats = set(self.filter_settings.get('ignore_names', [])) | \
                        set(self.filter_settings.get('category_names', []))
        
        for anno in self.dataset['annotations']:
            im_height = im_height_map.get(anno['image_id'], 0)
            ignore = is_ignore_annotation(anno, self.filter_settings, im_height)
            
            # 获取2D bbox
            bbox2D = None
            if self.filter_settings.get('trunc_2D_boxes') and 'bbox2D_trunc' in anno and \
               not np.all([val == -1 for val in anno['bbox2D_trunc']]):
                bbox2D = BoxMode.convert(anno['bbox2D_trunc'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            elif anno.get('bbox2D_proj', [-1])[0] != -1:
                bbox2D = BoxMode.convert(anno['bbox2D_proj'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            elif anno.get('bbox2D_tight', [-1])[0] != -1:
                bbox2D = BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            else:
                continue  # 没有有效的2D bbox
            
            # 添加额外字段
            anno['area'] = bbox2D[2] * bbox2D[3]
            anno['iscrowd'] = False
            anno['ignore'] = ignore
            anno['ignore2D'] = ignore
            anno['ignore3D'] = ignore
            
            if self.filter_settings.get('modal_2D_boxes') and anno.get('bbox2D_tight', [-1])[0] != -1:
                anno['bbox'] = BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            else:
                anno['bbox'] = bbox2D
            
            anno['bbox3D'] = anno.get('bbox3D_cam', [])
            anno['depth'] = anno.get('center_cam', [0, 0, 0])[2]
            
            # 检查类别
            category_name = anno.get('category_name', '')
            if category_name in trainable_cats:
                valid_anns.append(anno)
        
        self.dataset['annotations'] = valid_anns
    
    def _create_index(self):
        """创建索引"""
        # 图片索引
        for img in self.dataset['images']:
            self.images.append(img)
        
        # 标注索引
        for ann in self.dataset['annotations']:
            self.annotations.append(ann)
            self.imgToAnns[ann['image_id']].append(ann)
            self.catToImgs[ann['category_id']].append(ann['image_id'])
    
    def get_image_ids(self):
        """获取所有图片ID"""
        return [img['id'] for img in self.images]
    
    def get_annotation_ids(self, img_ids=None, cat_ids=None):
        """获取标注ID"""
        if img_ids is None and cat_ids is None:
            return [ann['id'] for ann in self.annotations]
        
        ann_ids = []
        for ann in self.annotations:
            if img_ids is not None and ann['image_id'] not in img_ids:
                continue
            if cat_ids is not None and ann['category_id'] not in cat_ids:
                continue
            ann_ids.append(ann['id'])
        
        return ann_ids
    
    def load_images(self, img_ids):
        """加载图片信息"""
        img_dict = {img['id']: img for img in self.images}
        return [img_dict[img_id] for img_id in img_ids if img_id in img_dict]
    
    def load_annotations(self, ann_ids):
        """加载标注信息"""
        ann_dict = {ann['id']: ann for ann in self.annotations}
        return [ann_dict[ann_id] for ann_id in ann_ids if ann_id in ann_dict]
    
    def get_image_path(self, img):
        """
        获取图片的完整路径
        Args:
            img: 图片字典（包含file_path字段）
        Returns:
            完整图片路径
        """
        file_path = img.get('file_path', '')
        if not file_path:
            return None
        
        if self.data_root:
            return os.path.join(self.data_root, file_path)
        else:
            return file_path
    
    def get_image_annotations(self, img_id):
        """获取图片的所有标注"""
        return self.imgToAnns.get(img_id, [])
    
    def info(self):
        """打印数据集信息"""
        infos = self.dataset.get('info', {})
        if isinstance(infos, dict):
            infos = [infos]
        
        print("=" * 80)
        print("Dataset Information")
        print("=" * 80)
        for i, info in enumerate(infos):
            print(f'\nDataset {i+1}/{len(infos)}:')
            for key, value in info.items():
                print(f'  {key}: {value}')
        
        print(f'\nTotal Images: {len(self.images)}')
        print(f'Total Annotations: {len(self.annotations)}')
        print(f'Total Categories: {len(self.categories)}')
        print("=" * 80)
