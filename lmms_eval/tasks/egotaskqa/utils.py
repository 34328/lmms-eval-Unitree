import os  
import time  
import yaml  
from pathlib import Path

from openai import OpenAI    
from loguru import logger as eval_logger  

from lmms_eval.llm_judge.protocol import ServerConfig  
from lmms_eval.llm_judge import get_server  
  
# 初始化 OpenAI 客户端  
# API_KEY = os.getenv("OPENAI_API_KEY", "bce-v3/ALTAK-tnSvoDJzbcjn9wFinrEOm/e31cebbf2e9d7256e6b4fd38b90f96422f1d036b")  
# client = OpenAI(api_key=API_KEY, base_url="https://qianfan.baidubce.com/v2") 
API_TYPE = os.getenv("API_TYPE", "openai")  
VIDEO_DIR = os.getenv("QA_VIDEO_PATH", "/home/unitree/桌面/datasets/qa_videos")  
  
  
server_config = ServerConfig(  
    model_name="qwen3-235b-a22b-instruct-2507",  
    temperature=0.0,  
    max_tokens=5  
)  
server = get_server(server_name=API_TYPE, config=server_config) 




# 加载默认配置模板  
with open(Path(__file__).parent / "_default_template_yaml", "r") as f:  
    raw_data = f.readlines()  
    safe_data = []  
    for i, line in enumerate(raw_data):  
        if "!function" not in line:  
            safe_data.append(line)  
    config = yaml.safe_load("".join(safe_data))  
  
    
def egotaskqa_doc_to_visual(doc):  
    video_name = doc.get("interval")  
    video_path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")  
    if not os.path.exists(video_path):  
        raise FileNotFoundError(f"Video not found: {video_path}")  
    return [video_path]


def egotaskqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):  
    """  
    格式化 ERQA 数据集中的问题和选项。  
    """  
    if lmms_eval_specific_kwargs is None:  
        lmms_eval_specific_kwargs = {}  
      
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")  

    question = doc["question"]  
    # print("is Sample Text")
      
    return f"{pre_prompt}{question}"  
  
  
  
def egotaskqa_process_results(doc, result):  
    if not result or len(result) == 0:  
        return {"acc": {"question_type": doc.get("type", "unknown"), "correct": 0}}  
    # print('pred:',result)
    pred = result[0].strip()  
    is_correct = check_semantic_consistency(pred, doc)

    # 处理 type 可能是列表的情况  
    qtype = doc.get("type", ["unknown"]) 
    if not isinstance(qtype, list):    
        qtype = [str(qtype)]  
    elif len(qtype) == 0:  
        qtype = ["unknown"]  

    qtype_key = tuple(sorted(qtype)) 
    return {"acc": {"question_type": qtype_key, "correct": int(is_correct)}}

  
def egotaskqa_aggregate_accuracy(results):  
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
        # 将元组转换为可读的字符串  
        qtype_str = ", ".join(qtype)  
        eval_logger.info(f"Question Type: [{qtype_str}]: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")     
      
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
        qtype_str = ", ".join(qtype)  # 转换为可读字符串  
        table_lines.append(f"| [{qtype_str}] | {accuracy:.2f}% | {stats['correct']} | {stats['total']} |")    
        
    # 添加总计行    
    table_lines.append("|---------------|----------|---------|-------|")    
    table_lines.append(f"| Overall | {overall_accuracy:.2f}% | {total_correct} | {total_count} |")    
        
    # 输出表格    
    for line in table_lines:    
        eval_logger.info(line)    
        
    eval_logger.info("=" * 60 + "\n")    
        
    return overall_accuracy



def check_semantic_consistency(pred, doc):  
    """使用 OpenAI API 判断预测答案和标准答案的语义一致性"""  
    messages = f"""
        You are an AI assistant tasked with evaluating whether a response matches the correct answer to a given question, considering both \
            the primary answer and any extra correct answers.

        ## Evaluation Rules
        (1) Output 1 if the response matches the answer or any of the extra answers exactly or with synonymous/equivalent wording.
        - Synonyms, paraphrases, or different surface forms of the same meaning count as matches.
        - Minor wording differences (e.g., “Wood panel” vs. “Wood) count as matches.
        (2) Output 0 if the response is incorrect, contradictory, or refers to a different entity,object, or attribute than the answer and all extra answers.
        - If the answer and response describe different objects, actions, or states, mark as 0.
        - If the response introduces additional details that change the meaning of the answer, mark as 0.

        ##Special Cases
        - Similar meaning: Output 1 if the response conveys essentially the same meaning as the answer and does not omit or add critical \
            information (e.g., answer: “A ceiling fan”, response: “fan”).
        - Partial matches: If the response overlaps but misses or alters essential details (e.g.,answer: “put meat and tomato on the table” vs.\
              response: “put meat on the table”), output 0.
        - Granularity differences: If the response is more specific but still semantically equivalent (e.g., answer: “woman”, response: “Jessica”), output 1.
        - Yes/No questions: Only output 1 if the polarity matches (yes <-> yes, no <-> no). Any  mismatch outputs 0, regardless of explanation.
        - Ambiguity: If the response cannot be reasonably interpreted as equivalent to the answer,output 0.

        ## Examples:
        Example 1:
        Question: Is it overcast?
        Answer: no
        Extra Answers: ["doesn’t look like it", "no", "it’s sunny"]
        Response: yes
        Your output: 0

        Example 2:
        Question: Who is standing at the table?
        Answer: woman
        Extra Answers: ["a woman", "a lady", "woman"]
        Response: Jessica
        Your output: 1

        Example 3:
        Question: Are there drapes to the right of the bed?
        Answer: yes
        Extra Answers: ["yes, there are drapes", "yeah", "the drapes are to the right of the king bed"]
        Response: yes
        Your output: 1

        Example 4:
        Question: What material is the ceiling in the living room?
        Answer: Wood panel
        Extra Answers: null
        Response: wood
        Your output:1

        Example 5:
        Question: What is in between the two picture frames on the blue wall in the living room?
        Answer: The TV
        Extra Answers: null
        Response: air conditioner
        Your output: 0

        Example 6:
        Question: Is the house doorway open or closed?
        Answer: Open
        Extra Answers: null
        Response: The house doorway is open.
        Your output: 1

        Example 7:
        Question: Is my backyard safe to let me dog out in?
        Answer: Yes, its fenced.
        Extra Answers: null
        Response: yes
        Your output: 1

        Example 8:
        Question: What is hanging from the ceiling in the bedroom?
        Answer: A ceiling fan
        Extra Answers: null
        Response: fan
        Your output: 1

        Example 9:
        Question: Where is the full body mirror?
        Answer: In the bedroom by the door
        Extra Answers: ["next to the bedroom door", "just inside the bedroom", "in the bedroom", "in
        the bedroom right next to the door"]
        Response: The full body mirror is in the bedroom.
        Your output: 1

        Example 10:
        Question: What is leaning in the corner by the coat rack?
        Answer: An umbrella
        Extra Answers: null
        Response: chair
        Your output: 0

        Your Turn:
        Question: {doc["question"]}
        Answer: {doc["answer"]}
        Response: {pred}

"""
    time.sleep(2)
    try:  
        # 使用 llm_judge API 进行二元评估  
        result = server.evaluate_binary(  
            question = doc["question"],  
            answer = doc["answer"],  
            prediction=pred,  
            output_format="0/1",  
            custom_prompt=messages  
        )  
          
        # 解析结果  
        if result["success"]:  
            judge_response = result["result"]  
            judge_score = judge_response.strip() if isinstance(judge_response, str) else str(judge_response)  
            return 1 if judge_score == "1" else 0  
        else:  
            eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")  
            return 0  
              
    except Exception as e:  
        eval_logger.error(f"Error during judge evaluation: {str(e)}")  
        return 0