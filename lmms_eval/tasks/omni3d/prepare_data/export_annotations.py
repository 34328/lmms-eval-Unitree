#!/usr/bin/env python3
"""
将Omni3D数据导出为指定格式的JSON文件
参考格式：frame-000000.color.jpg.annotations.json

使用方法:
    # 基本导出（不生成可视化）
    python export_annotations.py --json path/to/Omni3D/KITTI_train.json \
                                  --data_root /path/to/datasets/ \
                                  --output_dir ./exported_annotations/ \
                                  --num_images 10
    
    # 导出并生成可视化图像（用于debug）
    python export_annotations.py --json path/to/Omni3D/KITTI_train.json \
                                  --data_root /path/to/datasets/ \
                                  --output_dir ./exported_annotations/ \
                                  --num_images 10 \
                                  --visualize
    
    # 指定可视化图像输出目录
    python export_annotations.py --json path/to/Omni3D/KITTI_train.json \
                                  --data_root /path/to/datasets/ \
                                  --output_dir ./exported_annotations/ \
                                  --vis_output_dir ./visualizations/ \
                                  --visualize
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
import shutil
from scipy.spatial.transform import Rotation as R_scipy

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入数据加载器
try:
    from load_omni3d_data import Omni3DLoader, BoxMode
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保load_omni3d_data.py在同一目录下")
    sys.exit(1)

# 导入可视化函数（可选）
try:
    from visualize_bbox import (
        draw_2d_bbox_simple, draw_3d_bbox_simple, draw_coordinate_axes,
        get_cuboid_vertices_3d, project_3d_to_2d, clip_line_liang_barsky
    )
    HAS_VISUALIZATION = True
except ImportError as e:
    HAS_VISUALIZATION = False
    print(f"警告: 无法导入可视化模块: {e}")
    print("可视化功能将不可用")


def rotation_matrix_to_euler_xyz(R):
    """
    将旋转矩阵转换为欧拉角（弧度）
    使用XYZ顺序（对应cubercnn的ZYX顺序，但返回格式为[x, y, z]）
    
    Args:
        R: 3x3旋转矩阵
    
    Returns:
        [rx, ry, rz]: 欧拉角（弧度），对应x, y, z轴的旋转
    """
    R = np.array(R)
    if R.shape != (3, 3):
        raise ValueError(f"R应该是3x3矩阵，得到{R.shape}")
    
    # 使用scipy的Rotation类，顺序为'xyz'（内旋顺序）
    # 这对应cubercnn的ZYX外旋顺序（R_z @ R_y @ R_x）
    r = R_scipy.from_matrix(R)
    euler = r.as_euler('xyz', degrees=False)
    
    # 返回 [rx, ry, rz] 格式
    return list(euler)


def intrinsic_3x3_to_4x4(K):
    """
    将3x3内参矩阵转换为4x4格式（添加第四行和第四列）
    
    Args:
        K: 3x3相机内参矩阵
    
    Returns:
        4x4内参矩阵
    """
    K = np.array(K)
    if K.shape != (3, 3):
        raise ValueError(f"K应该是3x3矩阵，得到{K.shape}")
    
    K_4x4 = np.eye(4)
    K_4x4[:3, :3] = K
    return K_4x4.tolist()


def get_identity_extrinsic():
    """
    返回单位外参矩阵（4x4）
    由于Omni3D数据中没有外参，使用单位矩阵
    """
    return np.eye(4).tolist()


def convert_bbox_2d(bbox, mode_from, mode_to='XYXY'):
    """
    转换2D bbox格式
    
    Args:
        bbox: bbox坐标
        mode_from: 源格式（BoxMode常量或字符串）
        mode_to: 目标格式（'XYXY' 或 'XYWH'）
    """
    # 如果源格式和目标格式相同，直接返回
    if mode_from == BoxMode.XYXY_ABS and mode_to == 'XYXY':
        return bbox
    if mode_from == BoxMode.XYWH_ABS and mode_to == 'XYWH':
        return bbox
    
    # 进行格式转换
    if mode_to == 'XYXY':
        return BoxMode.convert(bbox, mode_from, BoxMode.XYXY_ABS)
    elif mode_to == 'XYWH':
        return BoxMode.convert(bbox, mode_from, BoxMode.XYWH_ABS)
    
    return bbox


def convert_bbox_3d(center_cam, dimensions, R_cam):
    """
    将Omni3D的3D bbox格式转换为目标格式
    
    Args:
        center_cam: [x, y, z] 中心点（相机坐标系）
        dimensions: [w, h, l] 尺寸
        R_cam: 3x3旋转矩阵
    
    Returns:
        [cx, cy, cz, sx, sy, sz, rx, ry, rz]: 9-DoF bbox参数
    """
    center_cam = np.array(center_cam)
    dimensions = np.array(dimensions)
    
    TRANSFORM_MATRIX = np.array([
        [1, 0, 0],    # X' = X (right stays right)
        [0, 0, 1],    # Y' = Z (forward from old depth)
        [0, -1, 0]    # Z' = -Y (up from old -down)
    ])
    center_cam = TRANSFORM_MATRIX @ center_cam
    R_cam = TRANSFORM_MATRIX @ np.array(R_cam) @ TRANSFORM_MATRIX.T

    # dimensions = [w, l, h]
    w, h, l = dimensions
    dimensions = [l, w, h]

    R_z_90 = R_scipy.from_euler('z', -90, degrees=True).as_matrix()
    R_cam = R_cam @ R_z_90
    dimensions = [dimensions[1], dimensions[0], dimensions[2]]

    # 提取欧拉角
    euler = rotation_matrix_to_euler_xyz(R_cam)
    
    # 组合为9-DoF格式：[cx, cy, cz, sx, sy, sz, rx, ry, rz]
    bbox_3d = list(center_cam) + list(dimensions) + euler
    
    return bbox_3d


def euler_xyz_to_rotation_matrix(euler):
    """
    将欧拉角转换为旋转矩阵（逆操作）
    
    Args:
        euler: [rx, ry, rz] 欧拉角（弧度）
    
    Returns:
        3x3旋转矩阵
    """
    r = R_scipy.from_euler('xyz', euler, degrees=False)
    return r.as_matrix()


def draw_single_bbox(image, bbox_data, intrinsic, color, thickness, label, 
                     draw_2d=True, draw_3d=True, draw_axes=True):
    """
    绘制单个bbox（2D和3D）
    
    Args:
        image: 输入图像 (H, W, 3) BGR格式
        bbox_data: 包含bbox信息的字典，格式：
            {
                "bbox": [x1, y1, x2, y2],  # 2D bbox (XYXY格式)
                "bbox_3d": [cx, cy, cz, sx, sy, sz, rx, ry, rz],  # 9-DoF 3D bbox
                "category": "category_name"
            }
        intrinsic: 3x3相机内参矩阵
        color: BGR颜色元组
        thickness: 线条粗细
        label: 标签文本
        draw_2d: 是否绘制2D bbox
        draw_3d: 是否绘制3D bbox
        draw_axes: 是否绘制坐标轴
    
    Returns:
        绘制后的图像
    """
    if not HAS_VISUALIZATION:
        return image
    
    K = np.array(intrinsic)
    if K.shape == (4, 4):
        K = K[:3, :3]  # 如果是4x4，提取3x3部分
    
    # 绘制2D bbox
    if draw_2d and 'bbox' in bbox_data:
        bbox_2d = bbox_data['bbox']
        # 转换为XYWH格式用于绘制
        x1, y1, x2, y2 = bbox_2d
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
        image = draw_2d_bbox_simple(image, bbox_xywh, color=color, 
                                   thickness=thickness, label=label)
    
    # 绘制3D bbox
    if draw_3d and 'bbox_3d' in bbox_data:
        bbox_3d = bbox_data['bbox_3d']
        if len(bbox_3d) == 9:
            # 解析9-DoF格式：[cx, cy, cz, sx, sy, sz, rx, ry, rz]
            # 注意：这些数据已经经过了坐标变换（在convert_bbox_3d中）
            center_cam = np.array(bbox_3d[:3])
            dimensions = np.array(bbox_3d[3:6])
            euler = bbox_3d[6:9]
            
            # 检查中心点是否在相机前方
            if center_cam[1] <= 0:
                return image  # 跳过在相机后方的bbox
            
            # 从欧拉角重建旋转矩阵
            R_cam = euler_xyz_to_rotation_matrix(euler)
            
            # 计算3D立方体顶点
            # 注意：center_cam, dimensions, R_cam 已经是转换后的格式
            # 与 visualize_bbox.py 中的逻辑一致
            vertices_3d = get_cuboid_vertices_3d(center_cam, dimensions, R_cam)
            
            # 需要将顶点转换回原始坐标系（与 visualize_bbox.py 中的逻辑一致）
            TRANSFORM_MATRIX = np.array([
                [1, 0, 0],    # X' = X (right stays right)
                [0, 0, 1],    # Y' = Z (forward from old depth)
                [0, -1, 0]    # Z' = -Y (up from old -down)
            ])
            # 应用逆变换（转回原始坐标系）
            vertices_3d = (TRANSFORM_MATRIX.T @ vertices_3d.T).T
            
            # 绘制3D bbox线框
            image = draw_3d_bbox_simple(image, vertices_3d, K, 
                                       color=color, thickness=thickness)
            
            # 绘制坐标轴
            if draw_axes:
                # 坐标轴也需要使用转换后的数据，但 draw_coordinate_axes 内部会处理
                # 我们需要传入转换后的 center_cam, R_cam, dimensions
                image = draw_coordinate_axes(
                    image, center_cam, R_cam, dimensions, K,
                    axis_length=None, thickness=thickness
                )
    
    return image


def map_image_path(old_path):
    """
    将旧路径映射到新路径
    
    Args:
        old_path: 旧路径（完整路径或相对路径）
    
    Returns:
        映射后的新路径
    """
    new_path = old_path
    
    # 判断是否包含ARKitScenes或arkitscenes，进行路径替换
    if 'ARKitScenes' in old_path or 'arkitscenes' in old_path:
        import re
        import os
        
        # 替换路径前缀
        new_path = new_path.replace('detany3d_dataset/datasets', 'EmbodiedScan')
        new_path = new_path.replace('ARKitScenes', 'arkitscenes')
        
        # 提取序列ID和文件名
        # 例如: .../Validation/41069021/312.358_00000463.jpg
        pattern = r'Validation/(\d+)/([^/]+)$'
        match = re.search(pattern, new_path)
        if match:
            seq_id = match.group(1)  # 序列ID，如 41069021
            old_filename = match.group(2)  # 旧文件名，如 312.358_00000463.jpg
            
            # 从旧文件名中提取时间戳（第一个下划线前的部分）
            # 例如: 312.358_00000463.jpg -> 312.358
            timestamp_match = re.match(r'([\d.]+)', old_filename)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                # 构建新文件名：序列id_时间戳.png
                new_filename = f'{seq_id}_{timestamp}.png'
                # 替换路径：添加 _frames/lowres_wide/ 目录并更新文件名
                new_path = re.sub(
                    pattern,
                    rf'Validation/\1/\1_frames/lowres_wide/{new_filename}',
                    new_path
                )
    
    return new_path


def update_image_info_from_file(img_info, img_path):
    """
    根据新图像文件的实际分辨率更新img_info
    
    Args:
        img_info: 原始图像信息字典
        img_path: 新图像路径（如果不存在，会尝试映射）
    
    Returns:
        更新后的img_info字典，如果图像不存在则返回None
    """
    import copy
    
    # 如果路径不存在，尝试映射
    if not os.path.exists(img_path):
        mapped_path = map_image_path(img_path)
        if os.path.exists(mapped_path):
            img_path = mapped_path
        
    
    if not os.path.exists(img_path):
        return None
    
    # 读取图像获取实际分辨率
    image = cv2.imread(img_path)
    if image is None:
        return None
    
    new_height, new_width = image.shape[:2]
    old_width = img_info.get('width', new_width)
    old_height = img_info.get('height', new_height)
    
    # 计算缩放比例
    scale_x = new_width / old_width
    scale_y = new_height / old_height
    
    # 创建更新后的img_info副本
    updated_info = copy.deepcopy(img_info)
    updated_info['width'] = new_width
    updated_info['height'] = new_height
    
    # 更新相机内参矩阵K
    # K格式: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    K = np.array(img_info.get('K', np.eye(3)))
    if K.shape == (3, 3):
        K_updated = K.copy()
        # 缩放fx和cx（x方向）
        K_updated[0, 0] *= scale_x  # fx
        K_updated[0, 2] *= scale_x  # cx
        # 缩放fy和cy（y方向）
        K_updated[1, 1] *= scale_y  # fy
        K_updated[1, 2] *= scale_y  # cy
        updated_info['K'] = K_updated.tolist()
    
    return updated_info


def export_image_annotations(loader, img_id, output_dir=None, visualize=False, vis_output_dir=None):
    """
    导出单张图片的标注为JSON格式
    
    Args:
        loader: Omni3DLoader实例
        img_id: 图片ID
        output_dir: 输出目录（可选）
        visualize: 是否生成可视化图像
        vis_output_dir: 可视化图像输出目录（如果为None，使用output_dir）
    
    Returns:
        dict: 标注字典
    """
    # 加载图片信息
    imgs = loader.load_images([img_id])
    if len(imgs) == 0:
        print(f"图片ID {img_id} 不存在")
        return None
    
    img_info = imgs[0]
    
    # 获取原始图像路径
    img_path = loader.get_image_path(img_info)
    
    # 如果路径不存在，尝试使用路径映射
    if img_path and not os.path.exists(img_path):
        # 处理nuScenes路径问题（特殊处理）
        if "nuScenes" in img_path:
            img_path = img_path.replace("nuScenes", "nuscenes/data")
        
        # 如果仍然不存在，尝试路径映射
        if not os.path.exists(img_path):
            mapped_path = map_image_path(img_path)
            if os.path.exists(mapped_path):
                img_path = mapped_path
                print(f"  路径映射: {loader.get_image_path(img_info)} -> {img_path}")
    
    # 如果是ARKitScenes数据集，根据新图像的实际分辨率更新图像信息
    if img_path and ('ARKitScenes' in img_path or 'arkitscenes' in img_path):
        updated_info = update_image_info_from_file(img_info, img_path)
        if updated_info is not None:
            old_size = (img_info['width'], img_info['height'])
            new_size = (updated_info['width'], updated_info['height'])
            print(f"  更新图像信息: {old_size} -> {new_size}")
            img_info = updated_info
    
    # 获取标注
    anns = loader.get_image_annotations(img_id)
    
    # 获取数据集名称（从loader的dataset info中）
    dataset_name = 'unknown'
    if hasattr(loader, 'dataset') and 'info' in loader.dataset:
        info = loader.dataset['info']
        if isinstance(info, list) and len(info) > 0:
            info = info[0]
        dataset_name = info.get('name', info.get('id', 'unknown'))
    
    # 构建输出字典（使用更新后的img_info）
    output = {
        "dataset": dataset_name,
        "scene_name": img_info.get('file_path', '').split('/')[0] if '/' in img_info.get('file_path', '') else '',
        "image_identifier": img_info.get('file_path', ''),
        "image_path": img_path if img_path else '',
        "image_size": [img_info['width'], img_info['height']],
        "visible_instance_ids": list(range(len(anns))),  # 简单的ID列表
        "camera_annotations": {
            "extrinsic": get_identity_extrinsic(),  # Omni3D没有外参，使用单位矩阵
            "intrinsic": intrinsic_3x3_to_4x4(img_info['K'])
        },
        "object_grounding": []
    }
    if not os.path.exists(output["image_path"]):
        return None
    
    # 转换每个对象的标注
    for idx, ann in enumerate(anns):
        category_name = ann.get('category_name', 'unknown')
        
        # 获取2D bbox（优先使用projected box）
        bbox_2d = None
        if 'bbox2D_proj' in ann and ann['bbox2D_proj'][0] != -1:
            bbox_2d = convert_bbox_2d(ann['bbox2D_proj'], BoxMode.XYXY_ABS, 'XYXY')
        elif 'bbox' in ann:
            bbox_2d = convert_bbox_2d(ann['bbox'], BoxMode.XYWH_ABS, 'XYXY')
        elif 'bbox2D_tight' in ann and ann['bbox2D_tight'][0] != -1:
            bbox_2d = convert_bbox_2d(ann['bbox2D_tight'], BoxMode.XYXY_ABS, 'XYXY')
        
        if ann.get("ignore", False):
            continue
        
        if bbox_2d is None:
            continue
        
        # 转换3D bbox
        bbox_3d = None
        if 'center_cam' in ann and 'dimensions' in ann:
            center_cam = ann['center_cam']
            dimensions = ann['dimensions']
            R_cam = ann.get('R_cam', np.eye(3))
            
            try:
                bbox_3d = convert_bbox_3d(center_cam, dimensions, R_cam)
            except Exception as e:
                print(f"  警告: 无法转换3D bbox (ID {ann.get('id', 'unknown')}): {e}")
                continue
        
        if bbox_3d is None:
            continue
        
        # 构建对象标注
        obj_ann = {
            "name": f"{category_name}_{idx}",  # 添加索引以区分同名对象
            "category": category_name,
            "bbox": [float(x) for x in bbox_2d],  # 确保是float类型
            "bbox_3d": [float(x) for x in bbox_3d]  # 确保是float类型
        }
        
        output["object_grounding"].append(obj_ann)
    
    # 保存到文件（如果指定了输出目录）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名（基于image_identifier，使用完整路径避免重复）
        img_identifier = img_info.get('file_path', f'img_{img_id}')
        
        # 移除扩展名
        img_identifier_no_ext = os.path.splitext(img_identifier)[0]
        
        # 将路径分隔符替换为下划线，确保文件名唯一
        # 例如: SUNRGBD/kv2/kinect2data/.../0000121 -> SUNRGBD_kv2_kinect2data_..._0000121
        img_name = img_identifier_no_ext.replace('/', '_').replace('\\', '_')
        
        # 如果路径太长，可以只使用最后几级目录+文件名
        # 但为了确保唯一性，使用完整路径
        output_file = os.path.join(output_dir, f"{img_name}.annotations.json")
        
        with open(output_file, 'w') as f:
            json.dump([output], f, indent=4)
        
        print(f"  保存到: {output_file}")
        
        # 复制图像文件（使用已经映射和验证过的img_path）
        if img_path and os.path.exists(img_path):
            new_image_path = output_file.replace(".json", ".jpg")
            shutil.copy(img_path, new_image_path)
            print(f"  复制图片到: {new_image_path}")
            
            # 更新 image_identifier 和 image_path 为新路径
            output["image_identifier"] = os.path.relpath(new_image_path, os.path.dirname(output_dir)) if output_dir else new_image_path
            output["image_path"] = new_image_path
            
            # 重新保存JSON文件（因为image_identifier和image_path已更新）
            with open(output_file, 'w') as f:
                json.dump([output], f, indent=4)
        else:
            print(f"  警告: 图片文件不存在，跳过复制: {img_path}")

        
        # 生成可视化图像（如果启用）
        if visualize and HAS_VISUALIZATION:
            vis_dir = vis_output_dir if vis_output_dir else output_dir
            os.makedirs(vis_dir, exist_ok=True)
            
            # 生成可视化文件名（使用相同的命名规则）
            img_identifier = img_info.get('file_path', f'img_{img_id}')
            img_identifier_no_ext = os.path.splitext(img_identifier)[0]
            img_name = img_identifier_no_ext.replace('/', '_').replace('\\', '_')
            vis_output_path = os.path.join(vis_dir, f"{img_name}_vis.jpg")
            
            try:
                # 读取图像（使用已经映射和验证过的img_path）
                if not img_path or not os.path.exists(img_path):
                    print(f"  警告: 图片文件不存在: {img_path}")
                    return output
                
                image = cv2.imread(img_path)
                if image is None:
                    print(f"  警告: 无法读取图片: {img_path}")
                    return output
                
                # 获取内参矩阵
                K = np.array(img_info['K'])
                
                # 定义颜色列表
                colors = [
                    (0, 255, 0),    # 绿色
                    (255, 0, 0),    # 蓝色
                    (0, 0, 255),    # 红色
                    (255, 255, 0),  # 青色
                    (255, 0, 255),  # 洋红
                    (0, 255, 255),  # 黄色
                    (128, 0, 128),  # 紫色
                    (255, 165, 0),  # 橙色
                ]
                
                # 绘制每个bbox
                for i, bbox_data in enumerate(output["object_grounding"]):
                    try:
                        category = bbox_data.get("category", "unknown")
                        label = f"{category}"
                        color = colors[i % len(colors)]
                        thickness = 2
                        
                        image = draw_single_bbox(
                            image, bbox_data, K, color, thickness, label,
                            draw_2d=False, draw_3d=True, draw_axes=True
                        )
                    except Exception as e:
                        print(f"  警告: 绘制bbox {i} ({bbox_data.get('category', 'unknown')}) 失败: {e}")
                
                # 保存可视化图像
                cv2.imwrite(vis_output_path, image)
                print(f"  可视化图像保存到: {vis_output_path}")
            except Exception as e:
                print(f"  警告: 生成可视化图像失败: {e}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description='导出Omni3D数据为指定格式的JSON文件')
    parser.add_argument('--json', type=str, required=True,
                       help='JSON标注文件路径')
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据根目录（datasets目录）')
    parser.add_argument('--output_dir', type=str, default='./exported_annotations',
                       help='输出目录')
    parser.add_argument('--num_images', type=int, default=None,
                       help='要导出的图片数量（None表示全部）')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='起始图片索引')
    parser.add_argument('--visualize', '--vis', action='store_true',
                       help='生成可视化图像用于debug确认数据')
    parser.add_argument('--vis_output_dir', type=str, default=None,
                       help='可视化图像输出目录（默认与输出目录相同）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("=" * 80)
    print("加载数据...")
    print("=" * 80)
    loader = Omni3DLoader(
        annotation_files=args.json,
        data_root=args.data_root,
        filter_settings=None
    )
    
    loader.info()
    print()
    
    # 获取图片ID
    img_ids = loader.get_image_ids()
    total_images = len(img_ids)
    
    if args.num_images is None:
        num_to_export = total_images - args.start_idx
    else:
        num_to_export = min(args.num_images, total_images - args.start_idx)
    
    print(f"总图片数: {total_images}")
    print(f"将导出: {num_to_export} 张图片（从索引 {args.start_idx} 开始）")
    print()
    
    # 导出图片
    print("=" * 80)
    print("开始导出...")
    print("=" * 80)
    
    for i in range(num_to_export):
        idx = args.start_idx + i
        if idx >= total_images:
            break
        
        img_id = img_ids[idx]
        
        print(f"\n[{i+1}/{num_to_export}] 处理图片 ID: {img_id}")
        
        # 导出
        export_image_annotations(
            loader, img_id,
            output_dir=args.output_dir,
            visualize=args.visualize,
            vis_output_dir=args.vis_output_dir
        )
    
    print("\n" + "=" * 80)
    print(f"完成！导出结果保存在: {args.output_dir}")
    if args.visualize:
        vis_dir = args.vis_output_dir if args.vis_output_dir else args.output_dir
        print(f"可视化图像保存在: {vis_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

