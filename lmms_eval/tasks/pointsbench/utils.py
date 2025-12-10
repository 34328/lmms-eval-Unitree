import os
import json
import time 
import csv
import numpy as np
from PIL import Image
from loguru import logger as eval_logger  

from lmms_eval.tasks._task_utils.unitree_eval_utils import *

model_id = os.getenv("MODEL_ID", "qwen3-vl")
dataset_path = "/home/unitree/桌面/datasets/Points-Bench/points-bench"

METADATA_CSV_PATH = os.path.join(dataset_path,"pixmo_metadata.csv" )
IMAGE_POINTS_MAP = None

"""
Points-Bench 的 steerable 任务需要模型根据图片上的参考点进行定位或推理, pixmo_metadata.csv 文件中包含了这个种类每张图片对应的参考点坐标信息。
这里是先转为相对坐标(0-1之间) 给prompt。
"""

def load_image_points_map():
    """
    懒加载函数：只在第一次调用时读取 CSV，避免重复 IO 操作。
    """
    global IMAGE_POINTS_MAP
    if IMAGE_POINTS_MAP is not None:
        return

    IMAGE_POINTS_MAP = {}
    if not os.path.exists(METADATA_CSV_PATH):
        eval_logger.error(f"Metadata file not found at {METADATA_CSV_PATH}. Steerable tasks will fail.")
        return

    try:
        with open(METADATA_CSV_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 解析 points 列的 JSON 字符串
                if row['points'] and row['points'] != '[]':
                    try:
                        IMAGE_POINTS_MAP[row['image_filename']] = json.loads(row['points'])
                    except json.JSONDecodeError:
                        pass
        eval_logger.info(f"Loaded metadata for {len(IMAGE_POINTS_MAP)} images.")
    except Exception as e:
        eval_logger.error(f"Error loading metadata csv: {e}")

def get_steerable_context(doc):
    """
    重构后的函数：根据 doc 信息生成 Steerable 任务的上下文提示。
    
    Args:
        doc (dict): lmms-eval 传入的数据项，包含 'image', 'category', 'image_filename'
        
    Returns:
        str: 构造好的提示信息 (包含换行符)，如果非 steerable 任务则返回空字符串。
    """
        
    category = doc["category"]
    filename = doc["image_filename"]
    
    # 0. 只有 steerable 类别才处理，其他直接返回空
    if category != "steerable":
        return ""
    
    # 1. 确保数据已加载
    if IMAGE_POINTS_MAP is None:
        load_image_points_map()
    
    # 2. 检查该图片是否有原始点数据
    if filename not in IMAGE_POINTS_MAP:
        return ""
    
    # 3.  输入相对坐标
    original_points = IMAGE_POINTS_MAP[filename]
    pixel_coords_str = []
    
    for point in original_points:
        # 核心转换逻辑
        p_x = int(point["x"] * 10.0)
        p_y = int(point["y"] * 10.0)
        pixel_coords_str.append(f"[{p_x},{p_y}]")
        
    # 5. 构造最终提示语
    if pixel_coords_str:
        points_str = ", ".join(pixel_coords_str)
        # 注意：这里加了前导换行符，以便拼接到 User Prompt 中
        return f"\nThe image contains reference points is  {points_str}.\n The query refers to this existing point."
        
    return ""




def pointsbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):  

    return f"{doc['user_input']}"  
  

def pointsbench_doc_to_messages(doc, lmms_eval_specific_kwargs=None):  
    if lmms_eval_specific_kwargs is None:  
        lmms_eval_specific_kwargs = {}  
      
    img_name = doc["image_filename"]  
    question = doc["user_input"]  
    category = doc["category"]  

    imgs_path = os.path.join(dataset_path, "selected_images")
    img_path = os.path.join(imgs_path, category, img_name)  

    # 这一步会自动处理 CSV 读取、坐标转换和字符串生成
    context_info = get_steerable_context(doc)
  
    if category == "counting":  
        text_prompt = (  
            f"{question}\n"  
            "Output the point coordinates in JSON format: [{\"point\": [x, y]}, ...].\n"  
        )  
    else:  
        text_prompt = (  
            f"{question}\n"  
            f"{context_info}\n"
            "Output the point coordinates in JSON format: [{\"point\": [x, y]}].\n"  
        )  
      
    messages = [  
        {  
            "role": "system",  
            "content": [  
                {"type": "text", "text": "You are a helpful assistant capable of precise visual grounding."}  
            ]  
        },  
        {  
            "role": "user",  
            "content": [  
                {"type": "image", "url": img_path},  
                {"type": "text", "text": text_prompt}  
            ]  
        }  
    ]  
      
    return messages


def pointsbench_doc_to_target(doc):
    return None


def pointsbench_process_results(doc, result):
    """  
    处理模型的输出结果。  
    """  
    if not result or len(result) == 0:  
        return {"acc": {"question_type": doc.get("category", "unknown"), "correct": 0}}

    coordinate_cfg = MODEL_COORDINATE_CONFIGS[model_id]

    # 模型输出结果  
    points = decode_json_points(result[0].strip()  )
    if points is None:  
        return {"acc": {"question_type": doc.get("category", "unknown"), "correct": 0}}

    # 获取mask 便于计算 point是不是在mask内
    mask_path = os.path.join(dataset_path, "selected_masks/selected_masks")
    mask = os.path.join(mask_path, doc["category"], doc["mask_filename"])  
    mask = np.array(Image.open(mask))/255
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8)

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
    
        
    return {"acc": {"question_type": doc.get("category", "unknown"), "correct": int(correct)}}

def pointsbench_aggregate_accuracy(results):  
    """  
    计算并返回准确率。  
    按 question_type 分类统计,并输出 markdown 表格。  
    """  
    if not results:  
        return 0.0  
      
    # 统计每个 question_type 的正确数和总数  
    question_type_stats = {}  
      
    for result in results:  
        qtype = result.get("question_type", "unknown")  
        correct = result.get("correct", 0)  
          
        if qtype not in question_type_stats:  
            question_type_stats[qtype] = {"correct": 0, "total": 0}  
          
        question_type_stats[qtype]["correct"] += correct  
        question_type_stats[qtype]["total"] += 1  
      
    # 计算每个类别的准确率  
    category_accuracies = {}  
    for qtype, stats in question_type_stats.items():  
        accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0  
        category_accuracies[qtype] = accuracy  
        eval_logger.info(f"Question Type: {qtype}: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")  
      
    # 计算总体准确率  
    total_correct = sum(stats["correct"] for stats in question_type_stats.values())  
    total_count = sum(stats["total"] for stats in question_type_stats.values())  
    overall_accuracy = (total_correct / total_count * 100) if total_count > 0 else 0  
      
    # 输出 markdown 表格  
    eval_logger.info("\n" + "=" * 60)  
    eval_logger.info("Points-Bench Results by Question Type:")  
    eval_logger.info("=" * 60)  
      
    # 表格头  
    table_lines = []  
    table_lines.append("| Question Type | Accuracy | Correct | Total |")  
    table_lines.append("|---------------|----------|---------|-------|")  
      
    # 按 question_type 排序并添加行  
    for qtype in sorted(question_type_stats.keys()):  
        stats = question_type_stats[qtype]  
        accuracy = category_accuracies[qtype]  
        table_lines.append(f"| {qtype} | {accuracy:.2f}% | {stats['correct']} | {stats['total']} |")  
      
    # 添加总计行  
    table_lines.append("|---------------|----------|---------|-------|")  
    table_lines.append(f"| **Overall** | **{overall_accuracy:.2f}%** | **{total_correct}** | **{total_count}** |")  
      
    # 输出表格  
    for line in table_lines:  
        eval_logger.info(line)  
      
    eval_logger.info("=" * 60 + "\n")  
      
    return overall_accuracy






