import os
import sys
import yaml 
import json
from pathlib import Path  

import numpy as np
from PIL import Image  

import torch
from loguru import logger as eval_logger  

# with open(Path(__file__).parent / "omni3d_arkitscenes.yaml", "r") as f:  
#     raw_data = f.readlines()  
#     safe_data = []  
#     for i, line in enumerate(raw_data):  
#         if "!function" not in line:  
#             safe_data.append(line)  
# cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"] 

# 设置自己omni 四个数据文件的根目录
data_root = os.getenv("DATA_ROOT","/home/unitree/桌面/datasets/omni3d/datasets")


def omni_doc_to_text(doc, lmms_eval_specific_kwargs=None):    
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

def omni_doc_to_visual(doc):

    image_path = doc["image_identifier"]  
    full_image_path = os.path.join(data_root, image_path)
    if not os.path.exists(full_image_path):  
        eval_logger.error(f"Image path: {full_image_path} does not exist")  
        return []  
    return [Image.open(full_image_path).convert("RGB")]


def omni_process_results(doc, result):
    pred = result[0]
    pred_bbox_3d = parse_bbox_3d_from_text(pred)
    predNums = len(pred_bbox_3d) # 预测的物体数量
    gtNums = len(doc["object_grounding"])

    # 获取pred 和gt 的8个顶点坐标
    pred_vertices_list = []
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
    





