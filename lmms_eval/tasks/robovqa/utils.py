import os  
import re
import time
import yaml 
import random

import requests  
from pathlib import Path   
from datasets import Dataset 
from loguru import logger as eval_logger  

from lmms_eval.llm_judge.protocol import ServerConfig  
from lmms_eval.llm_judge import get_server 

# 加载默认配置模板  
with open(Path(__file__).parent / "_default_template_yaml", "r") as f:  
    raw_data = f.readlines()  
    safe_data = []  
    for i, line in enumerate(raw_data):  
        if "!function" not in line:  
            safe_data.append(line)  
    config = yaml.safe_load("".join(safe_data))  

# 视频地址基础 URL
VIDEO_BASE_URL = "https://storage.googleapis.com/gdm-robovqa/videos/"  
API_TYPE = os.getenv("API_TYPE", "openai")  
  
  
server_config = ServerConfig(  
    model_name="qwen3-235b-a22b-instruct-2507",  
    temperature=0.0,  
    max_tokens=5  
)  
server = get_server(server_name=API_TYPE, config=server_config) 


# 这个数据集有点麻烦，一个样本对应一个视频和ID，但是有多个问题，所以要处理成任务列表的形式
class Task:
    """A class for handling tags and splits in a given task."""

    # Tags for default splitting, based on who is talking.
    PRED_STARTS = ['Robot:', 'Thought:', 'Action:']
    NOPRED_STARTS = ['User:', 'System:']

    # Tags surrounding all blocks needing to be predicted by the model.
    PRED_START = '<PRED>'
    PRED_END = '</PRED>'
    # Tags surrounding only binary answers, typically 'yes' and 'no'.
    PRED_ANSWER_BINARY_START = '<PRED:ANSWER:BINARY>'
    PRED_ANSWER_BINARY_END = '</PRED:ANSWER:BINARY>'
    # Tags surrounding all discrete answers coming from a limited set of classes,
    # e.g. 'yes', 'no', 'halfway there', 'done', '10s', etc.
    PRED_ANSWER_DISCRETE_START = '<PRED:ANSWER:DISCRETE>'
    PRED_ANSWER_DISCRETE_END = '</PRED:ANSWER:DISCRETE>'
    # Tags surrounding things that constitute an answer to a question,
    # the question may be asked by a user or by the model itself.
    PRED_ANSWER_START = '<PRED:ANSWER'
    PRED_ANSWER_END = '</PRED:ANSWER'
    # Tags that have any sort of short-content value
    PRED_ALL_START = '<PRED:'
    PRED_ALL_END = '</PRED:'

    TAGS_RE = r'(</*\w[:\w]*>)'

    def __init__(self, text):
        self.text = text

    def get_random_split(self, split_type='speaker'):
        splits = self.get_splits(split_type)
        return random.choice(splits)

    def get_splits(self, split_type='speaker'):
        """Returns a list of (source, target) split pairs."""
        if split_type == 'pred':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_START], end_tags=[self.PRED_END])
        elif split_type == 'binary':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_ANSWER_BINARY_START],
                end_tags=[self.PRED_ANSWER_BINARY_END])
        elif split_type == 'discrete':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_ANSWER_DISCRETE_START],
                end_tags=[self.PRED_ANSWER_DISCRETE_END])
        elif split_type == 'answer':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_ANSWER_START],
                end_tags=[self.PRED_ANSWER_END])
        elif split_type == 'A:':
            return self.get_splits_from_tags(start_tags=['A:'], end_tags=[])
        elif split_type == 'speaker':
            return self.get_splits_from_tags(
                start_tags=self.PRED_STARTS, end_tags=self.NOPRED_STARTS)
        elif split_type == 'all':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_ALL_START],
                end_tags=[self.PRED_ALL_END]
            )
        else:
            raise ValueError(f'Unknown split type: {split_type}')

    def get_splits_from_tags(self, start_tags, end_tags):
        """Returns a list of (source, target) split pairs given start/end tags."""
        # Find all the first positions of a start element.
        split_positions = []
        position = 0
        while position < len(self.text):
            # Find the next start tag given current position.
            start_position = self.find_next_tag(position, start_tags)
            if start_position is None:
                break
            # Then find the first end tag after this start tag.
            end_position = self.find_next_tag(start_position, end_tags)
            if end_position is None:
                end_position = len(self.text)
            split_positions.append((start_position, end_position))
            position = end_position + 1
        return self.get_splits_from_positions(split_positions)

    def get_splits_from_positions(self, split_positions):
        """Returns a list of (source, target) split pairs given split positions."""
        # Create splits.
        splits = []
        for (split_position, end_position) in split_positions:
            source = ''
            if split_position > 0:
                source = self.text[:split_position]
                source = self._remove_tags(source)
            target = self.text[split_position:end_position]
            target = self._remove_tags(target)
            splits.append((source, target))

        # If no splits are found, return entire text.
        if not splits:
            splits = [('', self.text)]

        return splits

    def find_next_tag(self, position, tags):
        tag_position = None
        lower_text = self.text.lower()
        for tag in tags:
            p = lower_text.find(tag.lower(), position)
            if p >= 0 and (tag_position is None or p < tag_position):
                tag_position = p
        return tag_position

    def _remove_tags(self, text):
        return re.sub(self.TAGS_RE, '', text)

    def remove_tags(self):
        self.text = self._remove_tags(self.text)

    def __str__(self):
        return self.text

class Tasks:
    """A class for handling and holding tasks information."""

    TASK_RE = r'(<task[:\w]*>)'
    RE_FLAGS = re.IGNORECASE

    def __init__(self, tasks_raw=None):
        # Contains all tasks for each task type in this Tasks collection
        # key: str, task type (tag)
        # value: list[str], question-answers which belong to this task type
        self.tasks_dict = {}
        self.tasks_list = []
        self.tasks_types = []
        self.tasks_raw = tasks_raw
        if tasks_raw is not None:
            self.add(tasks_raw)

    def add(self, tasks):
        self.add_from_text(tasks)

    def add_from_dict(self, tasks_dict):
        for name, tasks in tasks_dict.items():
            if name not in self.tasks_dict:
                self.tasks_dict[name] = []
            self.tasks_dict[name].extend(tasks)
            self.tasks_list.extend(tasks)
            self.tasks_types.extend([name] * len(tasks))

    def add_from_text(self, text):
        task_dict = self.text_to_dict(text)
        self.add_from_dict(task_dict)

    def text_to_dict(self, text):
        """Returns all tasks associated with this video."""
        # Split a serialized string into raw strings of individual tasks
        split = re.split(self.TASK_RE, text, flags=self.RE_FLAGS)[1:]
        # Construct a dict of
        # key: str, task type (tag)
        # value: list[str], question-answers which belong to this task type
        tasks_dict = {}
        i = 0
        while i < len(split) - 1:
            tag = split[i].strip()
            task = split[i+1].lstrip()
            if task:
                if tag not in tasks_dict:
                    tasks_dict[tag] = []
                tasks_dict[tag].append(task)
            i += 2
        return tasks_dict

    def __str__(self, show_tasks=True):
        s_parts = []
        s_parts.append(f'{len(self.tasks_dict.keys())} task types in {len(self)} tasks:\n')
        for key in sorted(self.tasks_dict):
            tasks = self.tasks_dict[key]
            s_parts.append(
                f'{key} ({len(tasks)} / {len(self)}, {100 * len(tasks) / float(len(self)):.1f}%)'
            )
            if show_tasks:
                s_parts.append(f'\n\t{str(tasks)}')
            s_parts.append('\n')
        return ''.join(s_parts)

    def detailed_str(self):
        s_parts = []
        s_parts.append(f'Raw input: {str(self.tasks_raw)}')
        s_parts.append(f'\n{self.__str__(show_tasks=True)}')
        return ''.join(s_parts)

    def get_stats(self):
        return self.__str__(show_tasks=False)

    def __len__(self):
        return len(self.tasks_list)

    def get_tasks_list(self):
        return self.tasks_list

    def get_tasks_types(self):
        return self.tasks_types

    def get_random_task(self):
        if not self.tasks_list:
            raise ValueError('Unexpected empty tasks list')
        return random.choice(self.tasks_list)



def _normalize_answer_label(answer: str) -> str:
    if answer is None:
        return ""
    # 去掉行首的单字母选项和紧随的分隔符，然后去除前后空白
    return re.sub(r'^\s*[A-Za-z]\s*[:\)\.]\s*', '', str(answer)).strip()

def extract_task_fields(task_tag):
    if task_tag.startswith('<') and task_tag.endswith('>'):
        task_content = task_tag[1:-1]
    else:
        task_content = task_tag
    
    parts = task_content.split(':')
    if len(parts) >= 3:
        return f"{parts[1]}:{parts[2]}"
    else:
        return task_tag

def fetch_question_answer(text):
    tasks = Tasks(text)
    results = []
    for i, (task_type, tasks) in enumerate(tasks.tasks_dict.items()):
        for task in tasks:
            t = Task(task)
            splits = t.get_splits('A:')
            for split in splits:
                question, answer = split
                question = question.strip()
                answer = answer.strip()
                results.append((i, task_type, question, answer))
    return results


def robovqa_process_docs(dataset):  
    """  
    将一个视频对应多个问题的数据展开为多个样本 ，同时增加任务种类等等
    """  
    expanded_docs = []  
    id = 0
    for doc in dataset:  
        video_filename = doc["video"]  # 视频文件名  t
        text = doc["text"]  # 包含多个问题和答案的文本
        
        qa_list = fetch_question_answer(text)

        # 为每个问题创建一个新的 doc  
        for (i, task_type, question, answer) in qa_list:
            new_doc = {  
                "id": id,
                "task_type": extract_task_fields(task_type),  # 任务种类
                "question": question,     # 问题
                "video": video_filename,  # 相同的视频   
                "answer": _normalize_answer_label(answer) # 对应答案  
            }
            expanded_docs.append(new_doc)  
            id += 1
      
    return Dataset.from_list(expanded_docs)

def robovqa_doc_to_visual(doc):  
    # VIDEO_BASE_URL = "https://storage.googleapis.com/gdm-robovqa/videos/"  
    # 获取缓存目录  
    HF_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/")  
    cache_dir = os.path.join(os.path.expanduser(HF_HOME), "robovqa", "videos")  
    os.makedirs(cache_dir, exist_ok=True)  
      
    # 构建完整的远程 URL  
    video_filename = doc["video"]  
    video_url = VIDEO_BASE_URL + video_filename  
      
    # 本地文件路径  
    local_video_path = os.path.join(cache_dir, video_filename)  
      
    # 如果文件不存在,从 URL 下载  
    if not os.path.exists(local_video_path):  
        try:  
            # eval_logger.info(f"Downloading video from {video_url} to {local_video_path}")  
            response = requests.get(video_url, stream=True, timeout=120)  
            response.raise_for_status()  
              
            with open(local_video_path, 'wb') as f:  
                for chunk in response.iter_content(chunk_size=8192):  
                    if chunk:  
                        f.write(chunk)  
              
            eval_logger.info(f"Successfully downloaded {video_filename}")  
        except Exception as e:  
            eval_logger.error(f"Failed to download video {video_url}: {e}")  
            raise  
      
    return [local_video_path]
  
def robovqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):  
    if lmms_eval_specific_kwargs is None:  
        lmms_eval_specific_kwargs = {}  
      
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")  
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")  

    return f"{pre_prompt}{doc['question']}{post_prompt}"
      
    
def robovqa_process_results(doc, result):  
    if not result or len(result) == 0:  
        return {"acc": {"question_type": doc.get("task_type", "unknown"), "correct": 0}}  
    # print('pred:',result)
    pred = result[0].strip()  
    is_correct = check_semantic_consistency(pred, doc)

    # 处理 type 可能是列表的情况  
    qtype = doc.get("task_type", ["unknown"]) 
    if not isinstance(qtype, list):    
        qtype = [str(qtype)]  
    elif len(qtype) == 0:  
        qtype = ["unknown"]  

    qtype_key = tuple(sorted(qtype)) 
    return {"acc": {"question_type": qtype_key, "correct": int(is_correct)}}
  
  
def robovqa_aggregate_accuracy(results):  
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
    eval_logger.info("Results by Question Type:")  
    eval_logger.info("=" * 60)  
      
    # 表格头  
    table_lines = []  
    table_lines.append("|       Question Type      | Accuracy | Correct | Total |")  
    table_lines.append("|--------------------------|----------|---------|-------|")  
      
    # 按 question_type 排序并添加行  
    for qtype in sorted(question_type_stats.keys()):  
        stats = question_type_stats[qtype]  
        accuracy = category_accuracies[qtype]  
        table_lines.append(f"| {qtype} | {accuracy:.2f}% | {stats['correct']} | {stats['total']} |")  
      
    # 添加总计行  
    table_lines.append("|--------------------------|----------|---------|-------|")  
    table_lines.append(f"|         Overall          | {overall_accuracy:.2f}% | {total_correct} | {total_count} |")  
      
    # 输出表格  
    for line in table_lines:  
        eval_logger.info(line)  
      
    eval_logger.info("=" * 60 + "\n")  
      
    return overall_accuracy


def check_semantic_consistency(pred, doc):  
    """使用 OpenAI API 判断预测答案和标准答案的语义一致性"""  
    messages = f"""
        You are an AI assistant tasked with evaluating whether a response matches the correct answer to a given question.

        Evaluation Rules
        (1) Output 1 if the response matches the answer exactly or with synonymous/equivalent wording.
        - Synonyms, paraphrases, or different surface forms of the same meaning count as matches.
        - Minor wording differences (e.g., “put tomato into fridge” vs. “the person is putting a tomato 
        in the fridge”) count as matches.
        (2) Output 0 if the response is incorrect, contradictory, or refers to a different entity, object, or attribute.
        - If the answer and response describe different objects, actions, or states, mark as 0.
        - If the response introduces additional details that change the meaning of the answer, mark as 0.

        Special Cases
        - Similar meaning: Output 1 if the response conveys essentially the same meaning as the answer and does not omit or add critical information \
            (e.g., answer:“put meat on the table”, response:“The person moved meat from the fridge to the counter.”).
        - Partial matches: If the response overlaps but misses or alters essential details (e.g., answer:“put meat and tomato on the table” vs. response:“put meat on the table”), output 0.
        - Granularity differences: If the response is more specific but still semantically equivalent(e.g., answer:“woman”, response:“Jessica”), output 1.
        - Yes/No questions: Only output 1 if the polarity matches (yes <-> yes, no <-> no). Any mismatch outputs 0, regardless of explanation.
        - Ambiguity: If the response cannot be reasonably interpreted as equivalent to the answer, output 0.
        
        Examples
        Example 1
        Question: Did the attribute of plant changed because of the action getting something from something?
        Answer: yes
        Response: Yes, the attribute of plant got watered from no to yes after the action getting something from something.
        Your output: 1
       
        Example 2 Question: what status of fork changed while the person do the first action did before he/she put something to something?
        Answer: cleanliness
        Response: fork was in drawer before the person put fork to sink.
        Your output: 0
        
        Example 3
        Question: What is the person doing before he/she close something?
        Answer: Put tomato to fridge
        Response: The person is putting a tomato in the fridge.
        Your output: 1
        
        Example 4
        Question: What is the first action the person did in the video?
        Answer: Work on sofa
        Response: The person pulled out a chair.
        Your output: 0

        Example 5
        Question: How did the person changed the spatial relationships of meat?
        Answer: Put meat to table
        Response: The person moved meat from the fridge to the counter.
        Your output: 1

        Example 6
        Question: what status of fridge changed while the person do the first action did after he/she point to something?
        Answer: openess
        Response: The fridge was closed before the person point to something, and after that the fridge changed to open.
        Your output: 1

        Example 7
        Question: which object changed its status when the person do the last action in the video?
        Answer: fork
        Response: spoon
        Your output: 0

        Example 8
        Question: What is the action that just happened?
        Answer: Place can in the tray
        Response: The person puts the can on the table.
        Your output: 0

        Example 9
        Question: current goal is: Please place the fruits in the bowl then place the kitchen supplies into the holder. \
            last 20 steps: 1. put white packet in the bowl 2. put white packet in the bowl 3. put yellow packet in the bowl \
            4. put blue packet in the bowl 5. put blue packet  in the bowl 6. put blue packet in the bowl 7. put yellow packet in the bowl. What’s the immediate next step?
        Answer: Put duster in the black stand
        Response: put brush in the holder
        Your output: 0

        Your Turn:
        Question: {doc["question"]}
        Answer: {doc["answer"]}
        Response: {pred}
        Your output:
"""
    time.sleep(0.5)
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


