import re 
import ast
import json
import time 

import numpy as np
from loguru import logger as eval_logger  
  

  
def refspatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):  
    """格式化问题文本"""  
    if lmms_eval_specific_kwargs is None:  
        lmms_eval_specific_kwargs = {}  
      
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")  
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")  
      
    return f"{pre_prompt}{doc['prompt']}{post_prompt}"  
  

def refspatial_doc_to_messages(doc, lmms_eval_specific_kwargs=None):

    time.sleep(0.2)  
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    messages = []
    user_content = []
    
   # 获取前后提示语  
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")  
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")  
    question = f"{pre_prompt}{doc['prompt']}{post_prompt}"  

    images = doc.get("image", []) 
    user_content.append({"type": "image", "url": images})  
      
    user_content.append({"type": "text", "text": question})  
    messages.append({"role": "user", "content": user_content})  
    return messages


def refspatial_doc_to_target(doc):
    return None


def refspatial_process_results(doc, result):  
    """处理单个样本的结果"""  
    if not result or len(result) == 0:  
        return {"acc": 0}    
    
    # 获取mask 便于计算 point是不是在mask内
    mask = np.array(doc.get("mask"))/255
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8)
    
    # 处理模型返回的结果
    points = decode_json_points(result[0].strip())
    if points is None:  
        return {"acc": 0} 
    points = absolute_to_relative_points(points, mask.shape[1], mask.shape[0])
    acc = 0.0
    if len(points) > 0:
        in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & \
                    (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        acc = np.concatenate([
            mask[points[in_range, 1], points[in_range, 0]],
            np.zeros(points.shape[0] - in_range.sum())
        ]).mean()
    
    correct = 1 if acc == 1.0 else 0

    # test_split = doc.get("_config", {}).get("test_split") 
    return  {"acc": correct}  

  

def decode_json_points(text: str):
    """Parse coordinate points from text format"""
    try:
        # 清理markdown标记
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        
        # 解析JSON
        data = json.loads(text)
        points = []
        labels = []
        
        for item in data:
            if "point_2d" in item:
                x, y = item["point_2d"]
                x_norm = x/ 1000.0 
                y_norm = y/ 1000.0
                points.append((x_norm, y_norm))
                
                # 获取label，如果没有则使用默认值
                label = item.get("label", f"point_{len(points)}")
                labels.append(label)
            else:
                return None
        
        return points
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def absolute_to_relative_points(points, width, height):  
    """将绝对坐标转换为相对坐标  
      
    Args:  
        points: 绝对坐标列表 [(x1, y1), (x2, y2), ...]  
        width: 图像宽度  
        height: 图像高度    
        scale: 输出坐标的缩放比例，默认1000（0-1000范围）  
      
    Returns:  
        相对坐标列表 [(x1_norm, y1_norm), (x2_norm, y2_norm), ...]  
    """  
    relative_points = []  
    for x, y in points:  
        x_norm = int(x * width) 
        y_norm = int(y * height)
        relative_points.append((x_norm, y_norm))  
    return np.array(relative_points)