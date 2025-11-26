import re 
import ast
import json
import time 
from loguru import logger as eval_logger  
  

  
def robospatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):  
    """格式化问题文本"""  
    if lmms_eval_specific_kwargs is None:  
        lmms_eval_specific_kwargs = {}  
      
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")  
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")  
      
    return f"{pre_prompt}{doc['question']}{post_prompt}"  
  

def robospatial_doc_to_messages(doc, lmms_eval_specific_kwargs=None):

    time.sleep(0.2)  
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    messages = []
    user_content = []
    
    # 获取问题文本和预处理提示语
    if doc['category']=="context":
        question = f"{filter_prompt_instructions(doc['question'])} Output the point coordinates in JSON format."
    else:
        question = doc['question']

    images = doc.get("img", []) 

    user_content.append({"type": "image", "url": images})  
      
    user_content.append({"type": "text", "text": question})  
    messages.append({"role": "user", "content": user_content})  
    return messages



def robospatial_process_results(doc, result):  
    """处理单个样本的结果"""  
    if not result or len(result) == 0:  
        return {"acc": {"split_type": doc.get("split", "unknown"), "correct": 0}}  

    if doc['category']=="context":
        pred = str(decode_json_points(result[0].strip())[0]) 
    else:
        pred = result[0].strip()
    
    answer = doc["answer"]  
    correct, is_binary, parsed_answer, is_parsable = evaluate_answer(answer, pred)
    # 如果不可解析,打印警告并标记  
    if not is_parsable:  
        eval_logger.warning(  
            f"Unparsable answer for question={doc.get('question', 'unknown')}: "  
        )  
      
    split_type = doc.get("category", "unknown")    
    return {  
        "acc": {  
            "split_type": split_type,   
            "correct": int(correct),  
            "is_parsable": is_parsable  # 添加可解析标记  
        }  
    }  
  
 
  
def robospatial_aggregate_results(results):  
    """聚合所有结果,按 split 类型分类统计"""  
    if not results:  
        return 0.0  
      
    # 过滤掉不可解析的样本  
    valid_results = [r for r in results if r.get("is_parsable", True)]  
    unparsable_count = len(results) - len(valid_results)  
      
    if unparsable_count > 0:  
        eval_logger.warning(  
            f"Filtered out {unparsable_count} unparsable samples "  
            f"({unparsable_count/len(results)*100:.2f}% of total)"  
        )  
      
    if not valid_results:  
        eval_logger.error("No valid parsable results to aggregate")  
        return 0.0  
      
    # 统计每个 split 的正确数和总数  
    split_stats = {}  
      
    for result in valid_results:  
        split_type = result.get("split_type", "unknown")  
        correct = result.get("correct", 0)  
          
        if split_type not in split_stats:  
            split_stats[split_type] = {"correct": 0, "total": 0}  
          
        split_stats[split_type]["correct"] += correct  
        split_stats[split_type]["total"] += 1  
      
    # 输出每个 split 的准确率  
    eval_logger.info("\n" + "=" * 60)  
    eval_logger.info("RoboSpatial-Home Results by Split:")  
    eval_logger.info("=" * 60)  
      
    table_lines = []  
    table_lines.append("| Split Type | Accuracy | Correct | Total |")  
    table_lines.append("|------------|----------|---------|-------|")  
      
    for split_type in sorted(split_stats.keys()):  
        stats = split_stats[split_type]  
        accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0  
        table_lines.append(f"| {split_type} | {accuracy:.2f}% | {stats['correct']} | {stats['total']} |")  
      
    # 计算总体准确率(这是当前子任务的准确率)  
    total_correct = sum(stats["correct"] for stats in split_stats.values())  
    total_count = sum(stats["total"] for stats in split_stats.values())  
    overall_accuracy = (total_correct / total_count) if total_count > 0 else 0  # 返回比例而非百分比  
      
    table_lines.append("|------------|----------|---------|-------|")  
    table_lines.append(f"| Overall | {overall_accuracy*100:.2f}% | {total_correct} | {total_count} |")  
    table_lines.append(f"| (Excluded unparsable) | - | - | {unparsable_count} |")  
      
    for line in table_lines:  
        eval_logger.info(line)  
      
    eval_logger.info("=" * 60 + "\n")  
      
    # 返回比例(0-1之间),框架会自动计算组平均  
    return overall_accuracy

def filter_prompt_instructions(prompt_text: str) -> str:
    """
    从提示词中过滤掉从 'Your answer should be' 开始的所有指令性文本。
    """
    # 定义分隔符
    separator = 'Your answer should be'
    filtered_text = prompt_text.split(separator, 1)[0].strip()
    
    return filtered_text
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
                x_norm = x / 1000.0
                y_norm = y / 1000.0
                points.append((x_norm, y_norm))
                
                # 获取label，如果没有则使用默认值
                label = item.get("label", f"point_{len(points)}")
                labels.append(label)
        
        return points, labels
        
    except Exception as e:
        print(f"Error: {e}")
        return [], []

"""
判断 context 下回答的坐标正确 的逻辑 基于官方评测脚本
https://github.com/chanhee-luke/RoboSpatial-Eval/blob/master/evaluation.py
"""

def evaluate_answer(ground_truth, generated_answer):
    """
    Evaluates if the generated answer is correct based on the ground truth.
    Returns a tuple of (is_correct, is_binary_answer, parsed_answer, is_parsable).
    """
    gen_answer = generated_answer.strip().lower()
    gt_lower = ground_truth.strip().lower()
    
    # Check if this is a binary yes/no question
    if gt_lower in ["yes", "no"]:
        is_binary = True
        is_gt_yes = (gt_lower == "yes")
        # Binary answers are always considered parsable if they contain text
        is_parsable = len(gen_answer) > 0
        if is_gt_yes:
            correct = gen_answer.startswith("yes")
        else:
            correct = gen_answer.startswith("no")
        return correct, is_binary, gen_answer, is_parsable
    else:
        # Numeric evaluation: ground_truth is a list of points defining a polygon
        is_binary = False
        parsed_answer = None
        is_parsable = False  # Default to not parsable until we successfully parse
        
        try:
            gt_polygon = ast.literal_eval(ground_truth)
            if not isinstance(gt_polygon, list) or len(gt_polygon) < 3:
                return False, is_binary, parsed_answer, is_parsable
            
            # Extract the first coordinate pair using regex
            # Look for patterns like (0.1,0.2) or (0.1, 0.2) or [0.1, 0.2] or [0.1,0.2]
            # This approach is more robust than trying to parse the entire list
            
            # Try to match tuple format (x,y) or (x, y)
            tuple_match = re.search(r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)', generated_answer)
            if tuple_match:
                try:
                    x = float(tuple_match.group(1))
                    y = float(tuple_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass  # Continue to other formats if float conversion fails
            
            # Try to match list format [x,y] or [x, y]
            list_match = re.search(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]', generated_answer)
            if list_match:
                try:
                    x = float(list_match.group(1))
                    y = float(list_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass  # Continue to other formats if float conversion fails
            
            # Fall back to the original approach but with extra safety
            try:
                # Extract the first list (text between square brackets) from generated_answer
                # Use a regex that can handle multi-line content
                match = re.search(r'\[(.*?)\]', generated_answer, re.DOTALL)
                if match is None:
                    return False, is_binary, parsed_answer, is_parsable
                
                # Add spaces after commas if not present (to help ast.literal_eval)
                list_content = match.group(1)
                list_content = re.sub(r',(\S)', r', \1', list_content)
                
                # Try to fix truncated tuples by adding closing parenthesis and brackets if needed
                list_content = list_content.strip()
                if list_content.endswith(','):
                    list_content = list_content[:-1]
                
                list_str = '[' + list_content + ']'
                
                # Try to parse the list directly
                try:
                    gen_val = ast.literal_eval(list_str)
                except (SyntaxError, ValueError):
                    # If direct parsing fails, try to extract just the first tuple
                    tuple_match = re.search(r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)', list_content)
                    if tuple_match:
                        x = float(tuple_match.group(1))
                        y = float(tuple_match.group(2))
                        parsed_answer = (x, y)
                        is_parsable = True
                        correct = point_in_polygon(x, y, gt_polygon)
                        return correct, is_binary, parsed_answer, is_parsable
                    else:
                        return False, is_binary, parsed_answer, is_parsable
                
                # Handle different formats for points
                if isinstance(gen_val, list):
                    if len(gen_val) == 0:
                        return False, is_binary, parsed_answer, is_parsable
                        
                    # Case 1: The list itself is a point coordinates [x, y]
                    if len(gen_val) == 2 and all(isinstance(v, (int, float)) for v in gen_val):
                        gen_point = tuple(gen_val)  # Convert [x, y] to (x, y)
                    # Case 2: The list contains points [(x, y), ...]
                    elif isinstance(gen_val[0], tuple):
                        gen_point = gen_val[0]
                    # Case 3: The list contains coordinate pairs as lists [[x, y], ...]
                    elif isinstance(gen_val[0], list) and len(gen_val[0]) == 2:
                        gen_point = tuple(gen_val[0])  # Convert [x, y] to (x, y)
                    else:
                        return False, is_binary, parsed_answer, is_parsable
                elif isinstance(gen_val, tuple):
                    gen_point = gen_val
                else:
                    return False, is_binary, parsed_answer, is_parsable

                if not (isinstance(gen_point, tuple) and len(gen_point) == 2):
                    return False, is_binary, parsed_answer, is_parsable
                
                x, y = float(gen_point[0]), float(gen_point[1])
                parsed_answer = (x, y)
                is_parsable = True
                correct = point_in_polygon(x, y, gt_polygon)
                return correct, is_binary, parsed_answer, is_parsable
            except Exception:
                # If all parsing attempts fail, return False
                return False, is_binary, parsed_answer, is_parsable
                
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return False, is_binary, parsed_answer, is_parsable
def point_in_polygon(x, y, poly):
    """
    Check if the point (x, y) lies within the polygon defined by a list of (x, y) tuples.
    Uses the ray-casting algorithm.
    """
    num = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, num + 1):
        p2x, p2y = poly[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside