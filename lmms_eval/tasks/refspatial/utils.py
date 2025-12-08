import os
import json
import time 

import numpy as np
from loguru import logger as eval_logger  

from lmms_eval.tasks._task_utils.unitree_eval_utils import *

model_id = os.getenv("MODEL_ID", "qwen3-vl")

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
    
    coordinate_cfg = MODEL_COORDINATE_CONFIGS[model_id]
    
    # 获取mask 便于计算 point是不是在mask内
    mask = np.array(doc.get("mask"))/255
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8)
    
    # 处理模型返回的结果
    points = decode_json_points(result[0].strip())
    if points is None:  
        return {"acc": 0} 
    
    if coordinate_cfg["is_relative"]: # 模型输出的是相对坐标  例如（500，800）
        points= relative_to_absolute_points(points, coordinate_cfg["default_image_size"]) # 先转为01之间的相对坐标
        points = np.array(absolute_to_relative_points(points, mask.shape)) # 再转为绝对坐标

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


