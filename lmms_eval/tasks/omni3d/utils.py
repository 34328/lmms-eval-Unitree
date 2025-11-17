import os
import sys
import time
import json
from pathlib import Path  

import numpy as np
from PIL import Image  
import cv2

import torch
from loguru import logger as eval_logger  
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.qhull import QhullError
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R_scipy
from lmms_eval.tasks.omni3d.prepare_data.visualize_bbox import get_cuboid_vertices_3d as get_cuboid_vertices_3d_gemini
from lmms_eval.tasks.omni3d.prepare_data.visualize_bbox import draw_3d_bbox_simple

# 设置自己omni 四个数据文件的根目录
data_root = os.getenv("DATA_ROOT","/home/unitree/桌面/datasets/omni3d/datasets")
model_id = os.getenv("MODEL_ID","") # 自己训练的模型和Qwen不是一个坐标系
SAVE_VISUAL_PATH = os.getenv("SAVE_VISUAL_PATH","")
if SAVE_VISUAL_PATH:
    os.makedirs(SAVE_VISUAL_PATH, exist_ok=True)


def omni3d_doc_to_text(doc, lmms_eval_specific_kwargs=None):    
    if lmms_eval_specific_kwargs is None:    
        lmms_eval_specific_kwargs = {}    
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")    
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")    
      
    # 从列表中提取所有物体的类别  
    object_grounding = doc.get("object_grounding", [])  
    categories = [obj["category"] for obj in object_grounding]  
      
    # 去重并格式化类别列表  
    unique_categories = list(set(categories))  # 去重  
    class_str = ", ".join(unique_categories)  # 用逗号连接  
      
    prompt = f"Locate the {class_str} in the provided image and output their positions and dimensions using 3D bounding boxes, The results must be in the JSON format: \
        `[{{\"bbox_3d\":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],\"label\":\"category\"}}]`."  
  
    return f"{pre_prompt}{prompt}{post_prompt}"

def omni3d_doc_to_target(doc):
    """  
    从 object_grounding 列表中提取所有对象的 bbox_3d  
    """  
    object_grounding = doc["object_grounding"]  
      
    # 提取所有对象的 bbox_3d  
    bbox_3d_list = [obj['bbox_3d'] for obj in object_grounding]  
      
    return bbox_3d_list
def omni3d_doc_to_visual(doc):
    # time.sleep(1.5)  # 避免触碰rpm
    image_path = doc["image_identifier"]  
    full_image_path = os.path.join(data_root, image_path)
    if not os.path.exists(full_image_path):  
        eval_logger.error(f"Image path: {full_image_path} does not exist")  
        return []  
    return [Image.open(full_image_path).convert("RGB")]


def visualize_bbox3d_comparison(doc, pred_vertices_list, gt_vertices_list, data_root, save_path):
    """
    可视化预测和GT的3D bbox对比，将两张图像拼接后保存
    
    Args:
        doc: 包含图像路径和相机内参的文档字典
        pred_vertices_list: 预测的3D bbox顶点列表，每个元素是8x3数组
        gt_vertices_list: GT的3D bbox顶点列表，每个元素是8x3数组
        data_root: 数据根目录
        save_path: 保存路径
    """
    try:
        # 获取图像路径
        image_path = doc.get("image_identifier", "")
        if not image_path:
            eval_logger.warning("无法获取图像路径，跳过可视化")
            return
        
        full_image_path = os.path.join(data_root, image_path)
        if not os.path.exists(full_image_path):
            eval_logger.warning(f"图像文件不存在: {full_image_path}，跳过可视化")
            return
        
        # 读取图像
        image = cv2.imread(full_image_path)
        if image is None:
            eval_logger.warning(f"无法读取图像: {full_image_path}，跳过可视化")
            return
        
        # 获取内参矩阵K
        camera_anns = doc.get("camera_annotations", {})
        intrinsic = camera_anns.get("intrinsic", None)
        if intrinsic is None:
            eval_logger.warning("无法获取相机内参，跳过可视化")
            return
        
        # 转换为numpy数组
        K = np.array(intrinsic)
        # 如果是4x4矩阵，提取3x3部分
        if K.shape == (4, 4):
            K = K[:3, :3]
        
        # 创建两个图像副本用于绘制
        image_pred = image.copy()
        image_gt = image.copy()
        
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
        
        # 绘制预测的bbox
        for i, vertices_3d in enumerate(pred_vertices_list):
            color = colors[i % len(colors)]
            try:
                image_pred = draw_3d_bbox_simple(
                    image_pred, vertices_3d, K, 
                    color=color, thickness=2
                )
            except Exception as e:
                eval_logger.warning(f"绘制预测bbox {i} 失败: {e}")
        
        # 绘制GT的bbox
        for i, vertices_3d in enumerate(gt_vertices_list):
            color = colors[i % len(colors)]
            try:
                image_gt = draw_3d_bbox_simple(
                    image_gt, vertices_3d, K, 
                    color=color, thickness=2
                )
            except Exception as e:
                eval_logger.warning(f"绘制GT bbox {i} 失败: {e}")
        
        # 添加标题文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        # 在预测图像上添加标题
        text_pred = "Prediction"
        (text_width, text_height), baseline = cv2.getTextSize(
            text_pred, font, font_scale, thickness
        )
        cv2.rectangle(
            image_pred, 
            (10, 10), 
            (10 + text_width + 10, 10 + text_height + baseline + 10),
            bg_color, -1
        )
        cv2.putText(
            image_pred, text_pred, (15, 10 + text_height + 5),
            font, font_scale, text_color, thickness
        )
        
        # 在GT图像上添加标题
        text_gt = "Ground Truth"
        (text_width, text_height), baseline = cv2.getTextSize(
            text_gt, font, font_scale, thickness
        )
        cv2.rectangle(
            image_gt, 
            (10, 10), 
            (10 + text_width + 10, 10 + text_height + baseline + 10),
            bg_color, -1
        )
        cv2.putText(
            image_gt, text_gt, (15, 10 + text_height + 5),
            font, font_scale, text_color, thickness
        )
        
        # 拼接两个图像（水平拼接）
        combined_image = np.hstack([image_pred, image_gt])
        
        # 生成保存文件名（使用完整路径避免冲突）
        image_name_no_ext = os.path.splitext(image_path)[0]
        # 处理路径中的特殊字符，替换为下划线
        safe_image_name = image_name_no_ext.replace('/', '_').replace('\\', '_')
        # 移除可能的其他特殊字符
        safe_image_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe_image_name)
        output_path = os.path.join(save_path, f"{safe_image_name}_vis.jpg")
        
        # 保存图像
        cv2.imwrite(output_path, combined_image)
        eval_logger.info(f"可视化图像已保存: {output_path}")
    except Exception as e:
        eval_logger.warning(f"可视化过程中出错: {e}")


def omni3d_process_results(doc, result):
    # 过滤后图片里面 没有样本  返回 None 会跳过这个样本  
    if not doc.get("object_grounding") or len(doc["object_grounding"]) == 0:  
        return {}  
    
    pred = result[0]
    pred_bbox_3d = parse_bbox_3d_from_text(pred)

    # 获取预测和真实的类别列表
    pred_categories = [item['label'] for item in pred_bbox_3d]
    gt_categories = [item['category'] for item in doc["object_grounding"]]

    # predNums = len(pred_bbox_3d) # 预测的物体数量
    # gtNums = len(doc["object_grounding"])

    # 验证：
    # print("预测的9DoF：",convert_normalized_angles_to_rad(pred_bbox_3d[0]["bbox_3d"]))
    # print("预测的9DoF：",convert_normalized_angles_to_rad(pred_bbox_3d[1]["bbox_3d"]))
    # print("真实的9DoF",doc["object_grounding"][0]["bbox_3d"])
    # print("真实的9DoF",doc["object_grounding"][1]["bbox_3d"])

    # 获取pred 和gt 的8个顶点坐标
    pred_vertices_list = []

    
    for item in pred_bbox_3d:
        center = item['bbox_3d'][:3]
        dimensions = item['bbox_3d'][3:6]
        # roll, pitch, yaw = item['bbox_3d'][6:9]
        pred_eular = convert_normalized_angles_to_rad(item['bbox_3d'])[-3:]
        rx, ry, rz = pred_eular[0], pred_eular[1], pred_eular[2]
        R_cam = euler_xyz_to_rotation_matrix(rx, ry, rz)
        
        vertices_3d = get_cuboid_vertices_3d(center, dimensions, R_cam)

        if "Qwen" not in model_id and model_id != "":
            TRANSFORM_MATRIX = np.array([
                [1, 0, 0],    # X' = X (right stays right)
                    [0, 0, 1],    # Y' = Z (forward from old depth)
                    [0, -1, 0]    # Z' = -Y (up from old -down)
                ])
                # 应用逆变换（转回原始坐标系）
            vertices_3d = (TRANSFORM_MATRIX.T @ vertices_3d.T).T
        pred_vertices_list.append(vertices_3d)
    
    gt_vertices_list = []
    for item in doc["object_grounding"]:
        if "bbox3d_cam" in item:
            gt_vertices_list.append(item.get("bbox3d_cam"))
        else:
            center = item['bbox_3d'][:3]
            dimensions = item['bbox_3d'][3:6]
            roll, pitch, yaw = item['bbox_3d'][6:9]
            R_cam = euler_xyz_to_rotation_matrix(roll, pitch, yaw)
            vertices_3d = get_cuboid_vertices_3d(center, dimensions, R_cam)
            TRANSFORM_MATRIX = np.array([
                [1, 0, 0],    # X' = X (right stays right)
                [0, 0, 1],    # Y' = Z (forward from old depth)
                [0, -1, 0]    # Z' = -Y (up from old -down)
            ])
            # 应用逆变换（转回原始坐标系）
            vertices_3d = (TRANSFORM_MATRIX.T @ vertices_3d.T).T
            gt_vertices_list.append(vertices_3d)
    
    # 可视化：如果设置了SAVE_VISUAL_PATH，保存预测和GT的bbox可视化
    if SAVE_VISUAL_PATH:
        visualize_bbox3d_comparison(doc, pred_vertices_list, gt_vertices_list, data_root, SAVE_VISUAL_PATH)
    
    # 1. 计算 IoU 矩阵
    iou_matrix = box3d_overlap_polyhedral(pred_vertices_list, gt_vertices_list)
    # 2. 最大二分图匹配 (仅基于 IoU)
    matches_iou, _, _ = match_boxes_by_iou(iou_matrix, iou_threshold=0.01)
    # 3. 增加类别匹配过滤
    final_matches = []
    for p_idx, g_idx, iou in matches_iou:
        # 类别匹配检查：只有 IoU 匹配，且类别也匹配时，才算作有效匹配
        if pred_categories[p_idx] == gt_categories[g_idx]:
            final_matches.append((p_idx, g_idx, iou))

    # 4. 准备计算 AP (使用过滤后的 final_matches)
    matches = final_matches
    
    # 确保每个 GT 框只被匹配一次
    gt_matched_flags = np.zeros(len(gt_vertices_list), dtype=bool)
    
    # 获取预测框的置信度 (这里假设 pred_bbox_3d 列表的顺序与 pred_vertices_list 一致)
    # 按照Qwen开发者Issues里面说的 置信度 假设其分数全为1.0
    pred_scores = np.ones(len(pred_vertices_list)) 

    # 按置信度从高到低排序 (使用预测列表的索引)
    sorted_idx = np.argsort(-pred_scores)
    
    tp = []
    fp = []
    
    for p_idx in sorted_idx:
        is_matched = False
        
        # 遍历所有有效的最终匹配，看是否有任何匹配涉及当前预测框 p_idx
        for match_p_idx, match_g_idx, iou in matches:
            # 必须满足 IoU 阈值 AND 类别已在 final_matches 中过滤 AND GT 框尚未被匹配
            if match_p_idx == p_idx and iou >= 0.15 and not gt_matched_flags[match_g_idx]:
                # 找到第一个合格匹配
                is_matched = True
                gt_matched_flags[match_g_idx] = True # 标记该 GT 框已被使用
                break
                
        if is_matched:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    # 5. 计算 AP15 (AP@0.15)
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    
    # 真实目标的数量 (作为召回率的分母)
    gt_count = len(doc["object_grounding"])
    
    if gt_count == 0:
        ap15 = 1.0 # 没有真实目标，如果没有预测框(fp=0)，AP=1
    elif len(tp) == 0:
        ap15 = 0.0 # 有真实目标，但没有预测框
    else:
        recall = tp / (gt_count + 1e-8)
        precision = tp / (tp + fp + 1e-8)
    
        # 11点插值法 (这里使用 101 个点插值，更精确)
        recall_points = np.linspace(0, 1, 101)
        
        # 计算插值后的最大精度
        precision_interp = [np.max(precision[recall >= r]) if np.any(recall >= r) else 0 for r in recall_points]
        ap15 = np.mean(precision_interp)

    # 返回指标字典  
    return {  
        "ap15": ap15  # 这个键名要与 YAML 中的 metric 名称对应  
    }




def omni3d_aggregate_results(results):  
    """  
    Args:  
        results: 一个列表,包含所有 process_results 返回的字典  
    Returns:  
        所有样本的平均 AP15 分数  
    """  
    if not results:  
        return 0.0  
      
    mean_ap15 = sum(results) / len(results)  
      
    eval_logger.info(f"Mean AP15: {mean_ap15:.4f}")  
    eval_logger.info(f"Total samples: {len(results)}")  
      
    return mean_ap15


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
    
    # 根据cubercnn/util/math_util.py中的实现：
    # X坐标：v0,v3,v4,v7是-l/2，v1,v2,v5,v6是+l/2
    # Y坐标：v0,v1,v4,v5是-h/2，v2,v3,v6,v7是+h/2  
    # Z坐标：v0,v1,v2,v3是-w/2，v4,v5,v6,v7是+w/2
    # 坐标系：+x right, +y down, +z toward screen
    
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

# vertices_3d = get_cuboid_vertices_3d(center_cam, dimensions, R_cam)

def parse_bbox_3d_from_text(text: str) -> list:
    """
    Parse 3D bounding box information from assistant response.
    
    Args:
        text: Assistant response text containing JSON with bbox_3d information
        
    Returns:
        List of dictionaries containing bbox_3d data
    """
    try:
        # Find JSON content
        if "```json" in text:
            start_idx = text.find("```json")
            end_idx = text.find("```", start_idx + 7)
            if end_idx != -1:
                json_str = text[start_idx + 7:end_idx].strip()
            else:
                json_str = text[start_idx + 7:].strip()
        else:
            # Find first [ and last ]
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx + 1]
            else:
                return []
        
        # Parse JSON
        bbox_data = json.loads(json_str)
        
        # Normalize to list format
        if isinstance(bbox_data, list):
            return bbox_data
        elif isinstance(bbox_data, dict):
            return [bbox_data]
        else:
            return []
            
    except (json.JSONDecodeError, IndexError, KeyError):
        return []
    
def convert_normalized_angles_to_rad(bbox_3d):
    """
    将归一化的角度值转换为弧度
    假设输入的最后三个值是相对于π的归一化值
    
    Args:
        bbox_3d: 包含9个元素的数组 [x, y, z, w, h, l, roll, pitch, yaw]
                最后三个值是归一化的角度值
    
    Returns:
        转换后的数组，最后三个值为弧度
    """
    # 复制数组避免修改原始数据
    result = list(bbox_3d)
    
    # 将最后三个值（归一化的角度）乘以180，然后转换为弧度
    for i in range(-3, 0):
        normalized_angle = bbox_3d[i]
        degrees = normalized_angle * 180
        radians = np.deg2rad(degrees)
        result[i] = radians
    
    return result


def convex_volume(points):
    """计算由点定义的凸包体积"""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except QhullError:
        return 0.0

def box3d_overlap_polyhedral(boxes_dt, boxes_gt):
    """
    手动计算3D IoU（支持旋转立方体）
    boxes_dt, boxes_gt: (N, 8, 3)
    输出: (N, M) 的IoU矩阵
    """
    boxes_dt = np.asarray(boxes_dt)
    boxes_gt = np.asarray(boxes_gt)
    N, M = boxes_dt.shape[0], boxes_gt.shape[0]
    ious = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        boxA = boxes_dt[i]
        try:
            hullA = ConvexHull(boxA)
            volA = hullA.volume
            eqA = hullA.equations  # Ax + b <= 0
        except QhullError:
            continue

        for j in range(M):
            boxB = boxes_gt[j]
            try:
                hullB = ConvexHull(boxB)
                volB = hullB.volume
                eqB = hullB.equations
            except QhullError:
                continue

            # 合并两组半空间
            halfspaces = np.vstack([eqA, eqB])
            interior_pt = (boxA.mean(axis=0) + boxB.mean(axis=0)) / 2

            try:
                hs = HalfspaceIntersection(halfspaces, interior_pt)
                inter_vol = ConvexHull(hs.intersections).volume
            except QhullError:
                inter_vol = 0.0
            except Exception:
                inter_vol = 0.0

            union_vol = volA + volB - inter_vol
            ious[i, j] = inter_vol / union_vol if union_vol > 0 else 0.0

    return ious

def match_boxes_by_iou(iou_matrix, iou_threshold=0.1):
    
    """
    根据 IoU 矩阵用匈牙利算法匹配预测框和GT框。

    Args:
        iou_matrix: numpy 数组，形状 (B1, B2)
        iou_threshold: 仅当 IoU >= 阈值 才视为有效匹配

    Returns:
        matches: [(pred_idx, gt_idx, iou)]
        unmatched_pred: list[int]
        unmatched_gt: list[int]
    """
    if not isinstance(iou_matrix, np.ndarray):
        iou_matrix = np.asarray(iou_matrix)

    B1, B2 = iou_matrix.shape
    # 匈牙利算法是最小化代价，而我们要最大化IoU => cost = -IoU
    cost_matrix = -iou_matrix

    # 调用 SciPy 内置匈牙利算法
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        iou = iou_matrix[i, j]
        if iou >= iou_threshold:
            matches.append((i, j, float(iou)))

    # 未匹配项
    matched_pred = {i for i, _, _ in matches}
    matched_gt = {j for _, j, _ in matches}
    unmatched_pred = [i for i in range(B1) if i not in matched_pred]
    unmatched_gt = [j for j in range(B2) if j not in matched_gt]

    return matches, unmatched_pred, unmatched_gt


def euler_xyz_to_rotation_matrix(rx, ry, rz):
    """
    将欧拉角 [rx, ry, rz] (弧度) 转换回 3x3 旋转矩阵 R。

    使用的约定与原有的 rotation_matrix_to_euler_xyz 函数严格一致：
    - 旋转顺序/约定：'xyz' (内旋顺序)
    - 角度单位：弧度 (radians)
    
    Args:
        rx (float): 绕 X 轴的旋转角 (弧度)。
        ry (float): 绕 Y 轴的旋转角 (弧度)。
        rz (float): 绕 Z 轴的旋转角 (弧度)。
        
    Returns:
        np.ndarray: 3x3 的旋转矩阵 R。
    """
    
    # 角度列表顺序必须与 'xyz' 顺序对应，即 [rx, ry, rz]
    euler_angles = [rx, ry, rz]
    
    # 使用 from_euler 方法，指定相同的 'xyz' 顺序和弧度单位
    r = R_scipy.from_euler(
        seq='xyz', 
        angles=euler_angles, 
        degrees=False  # 必须是 False，因为您原来的函数返回的是弧度
    )
    
    # 将 Rotation 对象转换为 3x3 矩阵
    return r.as_matrix()