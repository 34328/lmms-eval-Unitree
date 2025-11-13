import os
import sys
import yaml 
import json
from pathlib import Path  

import numpy as np
from PIL import Image  

import torch
from loguru import logger as eval_logger  
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.qhull import QhullError
from scipy.optimize import linear_sum_assignment

# 设置自己omni 四个数据文件的根目录
data_root = os.getenv("DATA_ROOT","/home/unitree/桌面/datasets/omni3d/datasets")


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

    image_path = doc["image_identifier"]  
    full_image_path = os.path.join(data_root, image_path)
    if not os.path.exists(full_image_path):  
        eval_logger.error(f"Image path: {full_image_path} does not exist")  
        return []  
    return [Image.open(full_image_path).convert("RGB")]


def omni3d_process_results(doc, result):
    # 过滤后图片里面 没有样本  返回 None 会跳过这个样本  
    if not doc.get("object_grounding") or len(doc["object_grounding"]) == 0:  
        return None  
    
    pred = result[0]
    pred_bbox_3d = parse_bbox_3d_from_text(pred)
    predNums = len(pred_bbox_3d) # 预测的物体数量
    gtNums = len(doc["object_grounding"])

    # 验证：
    # print("预测的9DoF：",convert_normalized_angles_to_rad(pred_bbox_3d[0]["bbox_3d"]))
    # print("预测的9DoF：",convert_normalized_angles_to_rad(pred_bbox_3d[1]["bbox_3d"]))
    # print("真实的9DoF",doc["object_grounding"][0]["bbox_3d"])
    # print("真实的9DoF",doc["object_grounding"][1]["bbox_3d"])

    # 获取pred 和gt 的8个顶点坐标
    pred_vertices_list = []
    # 每张图片的相机旋转矩阵 是固定的 选第一个就行
    R_cam =  doc["object_grounding"][0].get("R_cam", None)
    for item in pred_bbox_3d:
        center = item['bbox_3d'][:3]
        dimensions = item['bbox_3d'][3:6]
        roll, pitch, yaw = item['bbox_3d'][6:9]
        
        vertices_3d = get_cuboid_vertices_3d(center, dimensions, R_cam)
        pred_vertices_list.append(vertices_3d)
    gt_vertices_list = []
    for item in doc["object_grounding"]:
        gt_vertices_list.append(item.get("bbox3d_cam"))
    iou_matrix = box3d_overlap_polyhedral(pred_vertices_list, gt_vertices_list)
    # print(iou_matrix)
    matches, unmatched_pred, unmatched_gt = match_boxes_by_iou(iou_matrix, iou_threshold=0.01)

    pred_scores = np.ones(len(pred_vertices_list))  # 示例：全1；实际应为模型输出的置信度

    # 按置信度从高到低排序
    sorted_idx = np.argsort(-pred_scores)
    tp = []
    fp = []
    for i in sorted_idx:
        # 是否匹配上
        matched = any(m[0] == i and m[2] >= 0.15 for m in matches)
        if matched:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (len(gt_vertices_list) + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    # 计算 AP15
    recall_points = np.linspace(0, 1, 101)
    precision_interp = [np.max(precision[recall >= r]) if np.any(recall >= r) else 0 for r in recall_points]
    ap15 = np.mean(precision_interp)
    # print(f"当前样本 AP15 (基于matches) = {ap15:.4f}")
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
        [-l/2, -h/2, -w/2],  # v0: 左下后 (left-bottom-back)
        [ l/2, -h/2, -w/2],  # v1: 右下后 (right-bottom-back)
        [ l/2,  h/2, -w/2],  # v2: 右上后 (right-top-back)
        [-l/2,  h/2, -w/2],  # v3: 左上后 (left-top-back)
        [-l/2, -h/2,  w/2],  # v4: 左下前 (left-bottom-front)
        [ l/2, -h/2,  w/2],  # v5: 右下前 (right-bottom-front)
        [ l/2,  h/2,  w/2],  # v6: 右上前 (right-top-front)
        [-l/2,  h/2,  w/2],  # v7: 左上前 (left-top-front)
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