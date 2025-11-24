#!/usr/bin/env python3
"""
Omni3D数据可视化脚本
可视化2D bbox和3D bbox在图像上

使用方法:
    python visualize_bbox.py --json path/to/Omni3D/KITTI_train.json \
                              --data_root /path/to/datasets/ \
                              --output_dir ./output_vis/ \
                              --num_images 10
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
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

# 尝试导入cubercnn的可视化函数（可选）
try:
    from cubercnn import util, vis
    HAS_CUBERCNN_VIS = True
except ImportError:
    HAS_CUBERCNN_VIS = False
    print("警告: cubercnn可视化模块不可用，将使用简化版3D可视化")


def draw_2d_bbox_simple(img, bbox, color=(0, 255, 0), thickness=2, label=None):
    """
    简单的2D bbox绘制函数（不依赖cubercnn）
    Args:
        img: 图像数组 (H, W, 3)
        bbox: [x, y, w, h] 或 [x1, y1, x2, y2]
        color: BGR颜色元组
        thickness: 线条粗细
        label: 可选的标签文本
    """
    # 转换为numpy数组并确保是float类型
    bbox = np.array(bbox, dtype=np.float32).flatten()
    
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox format: {bbox}, expected 4 values")
    
    # 判断是XYWH还是XYXY格式
    # XYWH格式：w和h通常比较小（相对于图像尺寸）
    # XYXY格式：x2 > x1, y2 > y1，且差值较大
    x, y, w_or_x2, h_or_y2 = bbox
    
    # 简单判断：如果第三个值小于图像宽度的一半，且第四个值小于图像高度的一半，可能是XYWH
    # 或者如果第三个值明显大于第一个值（差值大于图像宽度的10%），可能是XYXY
    img_h, img_w = img.shape[:2]
    
    if w_or_x2 < img_w * 0.5 and h_or_y2 < img_h * 0.5:
        # XYWH格式：[x, y, w, h]
        x1, y1 = float(x), float(y)
        x2, y2 = float(x + w_or_x2), float(y + h_or_y2)
    elif w_or_x2 > x and h_or_y2 > y:
        # XYXY格式：[x1, y1, x2, y2]
        x1, y1, x2, y2 = float(x), float(y), float(w_or_x2), float(h_or_y2)
    else:
        # 默认当作XYWH处理
        x1, y1 = float(x), float(y)
        x2, y2 = float(x + w_or_x2), float(y + h_or_y2)
    
    # 确保坐标是整数且在有效范围内
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img_w - 1, int(x2)), min(img_h - 1, int(y2))
    
    # 检查bbox是否有效（宽度和高度都要大于0）
    if x2 <= x1 or y2 <= y1:
        return img  # 无效的bbox，跳过绘制
    
    # 绘制矩形
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # 绘制标签
    if label:
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # 绘制文本背景
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), color, -1)
        
        # 绘制文本
        cv2.putText(img, label, (x1, y1 - baseline - 2), 
                   font, font_scale, (255, 255, 255), text_thickness)
    
    return img


def get_cuboid_vertices_3d(center, dimensions, R=None):
    """
    计算3D立方体的8个顶点（相机坐标系）
    根据DATA.md中的定义：
    - 坐标系：+x right, +y down, +z toward screen
    - 顶点顺序：
                v4_____________________v5
                /|                    /|
               / |                   / |
              /  |                  /  |
             /___|_________________/   |
          v0|    |                 |v1 |
            |    |                 |   |
            |    |                 |   |
            |    |                 |   |
            |    |_________________|___|
            |   / v7               |   /v6
            |  /                   |  /
            | /                    | /
            |/_____________________|/
            v3                     v2
    
    Args:
        center: [x, y, z] 中心点（相机坐标系）
        dimensions: [w, h, l] 宽、高、长（米）
        R: 3x3旋转矩阵（可选）
    Returns:
        vertices: 8x3数组，8个顶点的3D坐标（按照v0-v7的顺序）
    """
    x, y, z = center
    w, h, l = dimensions

    
    # 相对于中心的8个顶点（局部坐标系）
    # 格式：[x, y, z]
    
    vertices_local = np.array([
        [-w/2, -h/2, -l/2],  # v0: 左下后 (left-bottom-back)
        [ w/2, -h/2, -l/2],  # v1: 右下后 (right-bottom-back)
        [ w/2,  h/2, -l/2],  # v2: 右上后 (right-top-back)
        [-w/2,  h/2, -l/2],  # v3: 左上后 (left-top-back)
        [-w/2, -h/2,  l/2],  # v4: 左下前 (left-bottom-front)
        [ w/2, -h/2,  l/2],  # v5: 右下前 (right-bottom-front)
        [ w/2,  h/2,  l/2],  # v6: 右上前 (right-top-front)
        [-w/2,  h/2,  l/2],  # v7: 左上前 (left-top-front)
    ])

    # 应用旋转
    if R is not None:
        R = np.array(R)
        vertices_local = (R @ vertices_local.T).T
    
    # 平移到中心位置
    vertices = vertices_local + np.array([x, y, z])
    
    return vertices


def project_3d_to_2d(vertices_3d, K):
    """
    将3D顶点投影到2D图像平面
    Args:
        vertices_3d: Nx3数组，3D顶点（相机坐标系）
        K: 3x3相机内参矩阵
    Returns:
        vertices_2d: Nx2数组，2D投影点
        valid_mask: N个布尔值，表示哪些点在相机前方
    """
    K = np.array(K)
    vertices_3d = np.array(vertices_3d)
    
    if vertices_3d.shape[1] != 3:
        raise ValueError(f"vertices_3d应该是Nx3数组，得到的是{vertices_3d.shape}")
    
    # 投影: [u, v, w] = K @ [x, y, z]
    vertices_2d_homogeneous = (K @ vertices_3d.T).T
    
    # 提取深度
    z = vertices_3d[:, 2]
    
    # 齐次坐标归一化
    vertices_2d = vertices_2d_homogeneous[:, :2] / np.maximum(vertices_2d_homogeneous[:, 2:3], 1e-6)
    
    # 检查有效性（在相机前方，z > 0）
    valid_mask = z > 0.1
    
    return vertices_2d, valid_mask


def clip_line_liang_barsky(p1, p2, width, height):
    """
    使用Liang-Barsky算法将线段裁剪到图像范围内
    Args:
        p1: 起点 (x, y)
        p2: 终点 (x, y)
        width: 图像宽度
        height: 图像高度
    Returns:
        裁剪后的线段 ((x1, y1), (x2, y2))，如果完全在图像外则返回None
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    
    dx = x2 - x1
    dy = y2 - y1
    
    # 参数化线段: (x, y) = (x1, y1) + t * (dx, dy), t in [0, 1]
    t_min, t_max = 0.0, 1.0
    
    # 检查四个边界：左(x=0), 右(x=width), 上(y=0), 下(y=height)
    for edge in range(4):
        if edge == 0:  # 左边界
            p, q = -dx, x1
        elif edge == 1:  # 右边界
            p, q = dx, width - 1 - x1
        elif edge == 2:  # 上边界
            p, q = -dy, y1
        else:  # 下边界
            p, q = dy, height - 1 - y1
        
        if abs(p) < 1e-6:
            # 线段平行于边界
            if q < 0:
                return None  # 完全在边界外
        else:
            t = q / p
            if p < 0:
                # 从外向内
                t_min = max(t_min, t)
            else:
                # 从内向外
                t_max = min(t_max, t)
            
            if t_min > t_max:
                return None  # 完全在图像外
    
    # 计算裁剪后的端点
    x1_clip = x1 + t_min * dx
    y1_clip = y1 + t_min * dy
    x2_clip = x1 + t_max * dx
    y2_clip = y1 + t_max * dy
    
    # 确保在图像范围内
    x1_clip = max(0, min(width - 1, int(round(x1_clip))))
    y1_clip = max(0, min(height - 1, int(round(y1_clip))))
    x2_clip = max(0, min(width - 1, int(round(x2_clip))))
    y2_clip = max(0, min(height - 1, int(round(y2_clip))))
    
    return ((x1_clip, y1_clip), (x2_clip, y2_clip))


def draw_3d_bbox_simple(img, vertices_3d, K, color=(0, 255, 0), thickness=2):
    """
    绘制3D bbox线框（参考提供的代码）
    Args:
        img: 图像数组 (H, W, 3) BGR格式
        vertices_3d: 8x3数组，3D立方体的8个顶点（相机坐标系）
        K: 3x3相机内参矩阵
        color: BGR颜色元组
        thickness: 线条粗细
    """
    if vertices_3d.shape[0] != 8:
        raise ValueError(f"vertices_3d应该有8个顶点，得到{vertices_3d.shape[0]}个")
    
    # 投影到2D
    vertices_2d, valid_mask = project_3d_to_2d(vertices_3d, K)
    
    if not valid_mask.any():
        print("所有点都在相机后方")
        return img  # 所有点都在相机后方
    
    h, w = img.shape[:2]
    
    # 定义12条边的连接关系（根据DATA.md中的顶点顺序）
    # v0-v1-v2-v3是后表面，v4-v5-v6-v7是前表面
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 后表面 (back face)
        [4, 5], [5, 6], [6, 7], [7, 4],  # 前表面 (front face)
        [0, 4], [1, 5], [2, 6], [3, 7],  # 连接前后表面的边
    ]
    
    # 绘制边（使用Liang-Barsky算法裁剪）
    for edge in edges:
        i, j = edge
        
        # 只绘制两个端点都在相机前方的边，避免投影错误
        if valid_mask[i] or valid_mask[j]:
            start_point = (vertices_2d[i][0], vertices_2d[i][1])
            end_point = (vertices_2d[j][0], vertices_2d[j][1])
            
            # 使用Liang-Barsky算法裁剪线段
            clipped = clip_line_liang_barsky(start_point, end_point, w, h)
            if clipped is not None:
                pt1_clip, pt2_clip = clipped
                # 只有当两个点不同时才绘制
                if pt1_clip != pt2_clip:
                    cv2.line(img, pt1_clip, pt2_clip, color, thickness)
    
    return img


def draw_coordinate_axes(img, center_cam, R_cam, dimensions, K, axis_length=None, thickness=2):
    """
    在bbox中心绘制坐标轴（X-红色，Y-绿色，Z-蓝色）
    
    Args:
        img: 输入图像 (H, W, 3) BGR格式
        center_cam: bbox中心点 [x, y, z]（相机坐标系）
        R_cam: 3x3旋转矩阵（从局部坐标系到相机坐标系）
        dimensions: bbox尺寸 [w, h, l]
        K: 3x3相机内参矩阵
        axis_length: 坐标轴长度，如果为None则使用bbox尺寸的平均值
        thickness: 线条粗细
        
    Returns:
        绘制后的图像
    """
    h, w = img.shape[:2]
    
    center_cam = np.array(center_cam)
    R_cam = np.array(R_cam)
    dimensions = np.array(dimensions)
    
    # 检查中心点是否在相机前方
    # if center_cam[2] <= 0:
    #     return img  # 中心点在相机后方，不绘制
    
    # 如果未指定轴长度，使用bbox尺寸的平均值
    if axis_length is None:
        axis_length = np.mean(dimensions) * 0.5
    
    # 定义坐标轴在局部坐标系中的方向（单位向量）
    axes_local = np.array([
        [axis_length, 0, 0],  # X轴
        [0, axis_length, 0],  # Y轴
        [0, 0, axis_length]   # Z轴
    ])
    
    # 将局部坐标轴转换到相机坐标系
    axes_cam = (R_cam @ axes_local.T).T + center_cam
    
    # 将中心点和坐标轴端点组合
    points_3d = np.vstack([center_cam.reshape(1, -1), axes_cam])
    TRANSFORM_MATRIX = np.array([
        [1, 0, 0],    # X' = X (right stays right)
        [0, 0, 1],    # Y' = Z (forward from old depth)
        [0, -1, 0]    # Z' = -Y (up from old -down)
    ])
    points_3d = (TRANSFORM_MATRIX.T @ points_3d.T).T
    
    # 投影到2D
    points_2d, valid_mask = project_3d_to_2d(points_3d, K)
    
    # 坐标轴颜色 (BGR格式)
    axis_colors = [
        (0, 0, 255),    # X轴 - 红色
        (0, 255, 0),    # Y轴 - 绿色
        (255, 0, 0)     # Z轴 - 蓝色
    ]
    
    # 绘制坐标轴
    if valid_mask[0]:  # 中心点在相机前方
        center_2d = points_2d[0]
        center_2d_int = (int(center_2d[0]), int(center_2d[1]))
        
        # 绘制中心点
        if 0 <= center_2d_int[0] < w and 0 <= center_2d_int[1] < h:
            cv2.circle(img, center_2d_int, 3, (255, 255, 255), -1)
        
        for i in range(3):
            if valid_mask[i + 1]:  # 轴端点在相机前方
                axis_end_2d = points_2d[i + 1]
                
                # 使用Liang-Barsky算法裁剪线段
                clipped = clip_line_liang_barsky(center_2d, axis_end_2d, w, h)
                if clipped is not None:
                    pt1_clip, pt2_clip = clipped
                    # 绘制轴线
                    if pt1_clip != pt2_clip:
                        cv2.line(img, pt1_clip, pt2_clip, axis_colors[i], thickness)
                    
                    # 在轴端点绘制箭头标记（如果在图像内）
                    end_point = (int(axis_end_2d[0]), int(axis_end_2d[1]))
                    if 0 <= end_point[0] < w and 0 <= end_point[1] < h:
                        cv2.circle(img, end_point, 4, axis_colors[i], -1)
    
    return img


def visualize_image_with_annotations(loader, img_id, output_path=None, show_3d=True):
    """
    可视化单张图片的2D和3D bbox
    Args:
        loader: Omni3DLoader实例
        img_id: 图片ID
        output_path: 输出路径（可选）
        show_3d: 是否显示3D bbox（需要pytorch3d）
    Returns:
        vis_img: 可视化后的图像
    """
    # 加载图片信息
    imgs = loader.load_images([img_id])
    if len(imgs) == 0:
        print(f"图片ID {img_id} 不存在")
        return None
    
    img_info = imgs[0]
    
    # 获取图片路径
    img_path = loader.get_image_path(img_info).replace("nuScenes", "nuscenes/data")
    if not img_path or not os.path.exists(img_path):
        print(f"图片文件不存在: {img_path}")
        return None
    
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return None
    
    # 转换为RGB（用于某些可视化函数）
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    img_bgr = img.copy()
    
    # 获取标注
    anns = loader.get_image_annotations(img_id)
    
    print(f"图片ID: {img_id}, 标注数量: {len(anns)}")
    
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
    
    # 准备3D可视化数据（用于cubercnn的高级可视化）
    meshes = []
    meshes_text = []
    
    # 绘制2D和3D bbox
    for idx, ann in enumerate(anns):
        category_name = ann.get('category_name', 'unknown')
        category_id = ann.get('category_id', -1)
        
        # 获取2D bbox（优先使用projected box）
        bbox_2d = None
        if 'bbox2D_proj' in ann and ann['bbox2D_proj'][0] != -1:
            bbox_xyxy = ann['bbox2D_proj']
            bbox_2d = BoxMode.convert(bbox_xyxy, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        elif 'bbox' in ann:
            bbox_2d = ann['bbox']
        elif 'bbox2D_tight' in ann and ann['bbox2D_tight'][0] != -1:
            bbox_xyxy = ann['bbox2D_tight']
            bbox_2d = BoxMode.convert(bbox_xyxy, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        
        if bbox_2d is None:
            continue
        
        # 选择颜色
        color_bgr = colors[idx % len(colors)]
        
        # 绘制2D bbox
        label = f"{category_name}"
        # img_bgr = draw_2d_bbox_simple(img_bgr, bbox_2d, color=color_bgr, 
        #                               thickness=2, label=label)
        
        # 绘制3D bbox
        # 优先使用bbox3D_cam（如果存在），否则从center和dimensions计算
        if show_3d:
            vertices_3d = None
            
            # # 方法1：直接使用bbox3D_cam（最准确）
            # if 'bbox3D_cam' in ann and ann['bbox3D_cam']:
            #     try:
            #         bbox3d_cam = np.array(ann['bbox3D_cam'], dtype=np.float32)
            #         if bbox3d_cam.shape == (8, 3):
            #             vertices_3d = bbox3d_cam  # 已经是8x3的顶点数组
            #     except Exception as e:
            #         pass
            
            # 方法2：从center和dimensions计算
            if vertices_3d is None and 'center_cam' in ann and 'dimensions' in ann:
                center_cam = ann['center_cam']
                dimensions = ann['dimensions']
                w, h, l = dimensions
                
                # 检查center_cam是否有效（不在相机后方）
                if isinstance(center_cam, (list, np.ndarray)) and len(center_cam) == 3:
                    if center_cam[2] <= 0:  # z <= 0 表示在相机后方或平面上
                        continue  # 跳过这个标注
                
                R_cam = ann.get('R_cam', np.eye(3))
                # T = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
                # T = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                # R_cam = T @ np.array(R_cam) @ T.T

                TRANSFORM_MATRIX = np.array([
                    [1, 0, 0],    # X' = X (right stays right)
                    [0, 0, 1],    # Y' = Z (forward from old depth)
                    [0, -1, 0]    # Z' = -Y (up from old -down)
                ])
                center_cam = TRANSFORM_MATRIX @ center_cam
                R_cam = TRANSFORM_MATRIX @ np.array(R_cam) @ TRANSFORM_MATRIX.T

                # dimensions = [w, l, h]
                dimensions = [l, w, h]

                R_z_90 = R_scipy.from_euler('z', -90, degrees=True).as_matrix()
                R_cam = R_cam @ R_z_90
                dimensions = [dimensions[1], dimensions[0], dimensions[2]]

                # R_cam = np.eye(3)
                vertices_3d = get_cuboid_vertices_3d(center_cam, dimensions, R_cam)
                vertices_3d = (TRANSFORM_MATRIX.T @ vertices_3d.T).T
                
            if vertices_3d is not None:
                # 绘制3D bbox线框
                K = np.array(img_info['K'])
                img_bgr = draw_3d_bbox_simple(img_bgr, vertices_3d, K, 
                                             color=color_bgr, thickness=2)
                
                # 绘制坐标轴
                center_cam = None
                dimensions = None
                R_cam = None
                
                # 优先使用annotation中的center_cam
                if 'center_cam' in ann and 'dimensions' in ann:
                    center_cam = ann['center_cam']
                    dimensions = ann['dimensions']
                    R_cam = ann.get('R_cam', np.eye(3))
                else:
                    # 如果没有center_cam，从vertices_3d计算中心点
                    # 但需要dimensions和R_cam来计算坐标轴方向
                    if 'dimensions' in ann:
                        center_cam = np.mean(vertices_3d, axis=0).tolist()
                        dimensions = ann['dimensions']
                        R_cam = ann.get('R_cam', np.eye(3))
                
                # 绘制坐标轴

                TRANSFORM_MATRIX = np.array([
                    [1, 0, 0],    # X' = X (right stays right)
                    [0, 0, 1],    # Y' = Z (forward from old depth)
                    [0, -1, 0]    # Z' = -Y (up from old -down)
                ])
                w, h, l = dimensions
                center_cam = TRANSFORM_MATRIX @ center_cam
                R_cam = TRANSFORM_MATRIX @ np.array(R_cam) @ TRANSFORM_MATRIX.T
                dimensions = [l, w, h]
                
                R_z_90 = R_scipy.from_euler('z', -90, degrees=True).as_matrix()
                
                R_cam = R_cam @ R_z_90
                # w, h = h, w
                dimensions = [dimensions[1], dimensions[0], dimensions[2]]
                img_bgr = draw_coordinate_axes(
                    img_bgr, center_cam, R_cam, dimensions, K,
                    axis_length=None, thickness=2
                            )
    
    # 保存或显示
    if output_path:
        cv2.imwrite(output_path, img_bgr)
        print(f"  保存到: {output_path}")
    
    return img_bgr


def main():
    parser = argparse.ArgumentParser(description='可视化Omni3D数据的2D和3D bbox')
    parser.add_argument('--json', type=str, required=True,
                       help='JSON标注文件路径')
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据根目录（datasets目录）')
    parser.add_argument('--output_dir', type=str, default='./output_vis',
                       help='输出目录')
    parser.add_argument('--num_images', type=int, default=10,
                       help='要可视化的图片数量')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='起始图片索引')
    parser.add_argument('--show', action='store_true',
                       help='显示图片（使用matplotlib）')
    parser.add_argument('--no_3d', action='store_true',
                       help='禁用3D可视化（仅显示2D bbox）')
    
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
    num_to_vis = min(args.num_images, total_images - args.start_idx)
    
    print(f"总图片数: {total_images}")
    print(f"将可视化: {num_to_vis} 张图片（从索引 {args.start_idx} 开始）")
    print()
    
    # 可视化图片
    print("=" * 80)
    print("开始可视化...")
    print("=" * 80)
    
    for i in range(num_to_vis):
        idx = args.start_idx + i
        if idx >= total_images:
            break
        
        img_id = img_ids[idx]
        
        # 生成输出文件名
        img_info = loader.load_images([img_id])[0]
        img_name = os.path.splitext(os.path.basename(img_info.get('file_path', f'img_{img_id}')))[0]
        output_path = os.path.join(args.output_dir, f'{img_name}_vis.jpg')
        
        print(f"\n[{i+1}/{num_to_vis}] 处理图片 ID: {img_id}")
        
        # 可视化
        vis_img = visualize_image_with_annotations(
            loader, img_id, 
            output_path=output_path,
            show_3d=not args.no_3d
        )
        
        # 显示（如果启用）
        if args.show and vis_img is not None:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 10))
                vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                plt.imshow(vis_img_rgb)
                plt.title(f'Image ID: {img_id}')
                plt.axis('off')
                plt.show(block=False)
                plt.pause(0.1)
            except Exception as e:
                print(f"  显示图片失败: {e}")
    
    print("\n" + "=" * 80)
    print(f"完成！可视化结果保存在: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

