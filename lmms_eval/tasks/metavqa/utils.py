import os  
import yaml  
import time
from pathlib import Path  
from loguru import logger as eval_logger  

from lmms_eval.tasks._task_utils.unitree_eval_utils import process_multiple_choice

dataset_path = "/home/unitree/桌面/datasets/metavqa/MetaVQA"

def metavqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):  
 
    if lmms_eval_specific_kwargs is None:  
        lmms_eval_specific_kwargs = {}  

    question = doc["question"]  
    return f"{question}"  
  

def metavqa_doc_to_messages(doc, lmms_eval_specific_kwargs=None):

    time.sleep(0.5)  # 防止日志输出混乱
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    messages = []
    user_content = []
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") 
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = f"{pre_prompt}{doc['question']}{post_prompt}"
    images_path = os.path.join(dataset_path, doc['obs'][0])
    user_content.append({"type": "image", "url": images_path})  
    # 添加文本  
    user_content.append({"type": "text", "text": question})  
    messages.append({"role": "user", "content": user_content})  
    return messages

def metavqa_doc_to_target(doc):  

    return doc.get("answer")  
  
  
def metavqa_process_results(doc, result):  
    """  
    处理模型的输出结果。  
    """  
    if not result or len(result) == 0:  
        return {"acc": {"question_type": doc.get("type", "unknown"), "correct": 0}}  
    # print('pred:',result)
    pred = process_multiple_choice(result[0].strip())  
    answer = process_multiple_choice(doc.get("answer", "") )
    is_correct = pred == answer    
        
    return {"acc": {"question_type": describe_re(doc.get("type")), "correct": int(is_correct)}}
  
  
def metavqa_aggregate_accuracy(results):  
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


def describe_re(s: str) -> str:
    return s.split('_', 1)[0]