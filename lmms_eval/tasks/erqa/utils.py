import os  
from pathlib import Path  
import yaml  
from loguru import logger as eval_logger  
from lmms_eval.tasks._task_utils.unitree_eval_utils import process_multiple_choice
  
# 加载默认配置模板  
with open(Path(__file__).parent / "_default_template_yaml", "r") as f:  
    raw_data = f.readlines()  
    safe_data = []  
    for i, line in enumerate(raw_data):  
        if "!function" not in line:  
            safe_data.append(line)  
    config = yaml.safe_load("".join(safe_data))  
  
  
def erqa_doc_to_visual(doc):  
    """  
    处理 ERQA 数据集中的图像。  
    返回图像列表供旧版接口使用。  
    """ 
    if "images" in doc and doc["images"]:  
        print("is Sample Img")
        return doc["images"]  
    elif "image" in doc:  
        print("is Sample Img")
        return [doc["image"]]  
    return []  
  
  
def erqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):  
    """  
    格式化 ERQA 数据集中的问题和选项。  
    """  
    if lmms_eval_specific_kwargs is None:  
        lmms_eval_specific_kwargs = {}  
      
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")  

    question = doc["question"]  
    # print("is Sample Text")
      
    return f"{pre_prompt}{question}"  
  

def erqa_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """
    将 ERQA 数据集转换为消息格式，支持根据 visual_indices 在文本中插入图片。
    
    Args:
        doc: 包含问题、图片和visual_indices的文档
        lmms_eval_specific_kwargs: 额外的参数配置
    
    Returns:
        list: 包含用户消息的列表，消息内容按照visual_indices排列图片和文本
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    messages = []
    user_content = []
    
    # 获取问题文本和预处理提示语
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    question = f"{pre_prompt}{doc['question']}"
    
    # 准备图片和位置索引对
    images = doc.get("images", []) if "images" in doc else [doc.get("image")] if "image" in doc else []
    visual_indices = doc.get("visual_indices", [])
    
    if not images:
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})
        return messages
    
    # 将图片和位置索引配对并排序
    image_index_pairs = list(zip(images, visual_indices))
    image_index_pairs.sort(key=lambda x: x[1])
    
    if not visual_indices or all(idx == 0 for idx in visual_indices):
        # 添加所有图片
        for img in images:
            user_content.append({"type": "image", "url": img})
        # 添加文本
        user_content.append({"type": "text", "text": question})
    else:
        # 根据索引位置在文本中插入图片
        last_pos = 0
        
        # 处理每个图片及其位置
        for img, idx in image_index_pairs:
            if idx == 0:
                # 图片放在开头
                user_content.append({"type": "image", "url": img})
            else:
                # 添加图片前的文本段
                if idx <= len(question):
                    text_segment = question[last_pos:idx]
                    if text_segment:
                        user_content.append({"type": "text", "text": text_segment})
                    user_content.append({"type": "image", "url": img})
                    last_pos = idx
                else:
                    user_content.append({"type": "image", "url": img})
        
        # 添加剩余的文本
        if last_pos < len(question):
            user_content.append({"type": "text", "text": question[last_pos:]})
    
    messages.append({"role": "user", "content": user_content})
    return messages
  

def erqa_doc_to_target(doc):  
    """  
    返回 ERQA 数据集中的正确答案。  
    """  
    return doc.get("answer", "")  
  
  
def erqa_process_results(doc, result):  
    """  
    处理模型的输出结果。  
    对于多项选择题,提取预测的选项字母并与答案比较。  
    """  
    if not result or len(result) == 0:  
        return {"acc": {"question_type": doc.get("question_type", "unknown"), "correct": 0}}  
    # print('pred:',result)

    answer = doc.get("answer", "")  
    pred = process_multiple_choice(result[0].strip())  
    answer = process_multiple_choice(doc.get("answer", "") )
        
    is_correct = pred == answer    
        
    return {"acc": {"question_type": doc.get("question_type", "unknown"), "correct": int(is_correct)}}
  
  
def erqa_aggregate_accuracy(results):  
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
    eval_logger.info("ERQA Results by Question Type:")  
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