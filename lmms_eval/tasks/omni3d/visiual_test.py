
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
import re
import sys
import argparse
import numpy as np
import cv2

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



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
        if valid_mask[i] and valid_mask[j]:
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


def visualize_image_with_annotations(img_path, anns, K, pred_vertices_list):
    """
    可视化单张图片的2D和3D bbox
    Args:
        img_path: 图片路径
        show_3d: 是否显示3D bbox（需要pytorch3d）
        filter_settings: 过滤设置
    Returns:
        vis_img: 可视化后的图像
    """
    
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return None
    
    # 转换为RGB（用于某些可视化函数）
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    img_bgr = img.copy()

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
    
    
    # 绘制2D和3D bbox
    valid_ann_count = 0
    for idx, ann in enumerate(anns):
        category_name = ann.get('category', 'unknown')

        color_bgr = colors[valid_ann_count % len(colors)]
        
        
        # 绘制3D bbox
        # 优先使用bbox3D_cam（如果存在），否则从center和dimensions计算
        if True:
            vertices_3d = None
            
            # 方法1：直接使用bbox3D_cam（最准确）
            if pred_vertices_list:
                try:
                    bbox3d_cam = np.array(pred_vertices_list[idx], dtype=np.float32)
                    if bbox3d_cam.shape == (8, 3):
                        vertices_3d = bbox3d_cam  # 已经是8x3的顶点数组
                except Exception as e:
                    pass
            
            
            if vertices_3d is not None:
                # 绘制3D bbox线框
                img_bgr = draw_3d_bbox_simple(img_bgr, vertices_3d, K, 
                                             color=color_bgr, thickness=2)
                
        
        valid_ann_count += 1
    
    match = re.search(r'(\d+_\d+\.(?:jpg|jpeg|png|bmp))', img_path, re.IGNORECASE)
    save_path = "/home/unitree/omni3d/"+ match.group(1).replace('.', '_3d_bboxes.')
    cv2.imwrite(save_path, img_bgr)
    # print(f"  图像已保存到: {save_path}")

    return img_bgr

