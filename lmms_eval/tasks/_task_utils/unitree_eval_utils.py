import re
import json
import numpy as np

def strip_answer(answer):
    answer = re.sub("The", "", answer)
    answer = re.sub("If", "", answer)
    answer = re.sub("[INST]", "", answer)
    answer = re.sub("[/INST]", "", answer)
    answer = re.sub("<Img>", "", answer)
    answer = re.sub("</Img>", "", answer)
    answer = answer.strip()
    return answer


def remove_special_characters(text):
    pattern = r"[-`\\【】\*\$、,，。.；;:：？\?！!\s\n\u4e00-\u9fff0-9①②③④⑤⑥⑦\[\]\<>a-z=\'\"\(\)\{\}]+"
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


def process_multiple_choice(answer):
    answer = strip_answer(answer)
    pattern = r"^([A-Z])\."
    matches = re.match(pattern, answer)
    if matches:
        return matches.group(1)
    key_words = [
        "boxed",
        "Answer:",
        "Answer is",
        "answer is",
        "option is",
        "Correct option",
        "correct option",
        "Answer",
        "answer",
        "故选",
        "选择",
        "正确选项为",
        "答案选",
        "答案为",
        "答案是",
        "因此",
        "答案",
    ]

    for key_word in key_words:
        if key_word in answer:
            answer = answer.split(key_word)[-1]
            break
    answer = remove_special_characters(answer)
    # keep the last line
    answer = answer.split("\n")[-1]
    pattern = r"[A-Z]"
    matches = re.findall(pattern, answer)
    return "".join(matches)


def absolute_to_relative_points(points, size):  
    """将相对坐标转换为图像中的绝对坐标  
      
    Args:  
        points: 相对坐标列表 [(x1, y1), (x2, y2), ...]  
        width: 图像宽度  
        height: 图像高度    
        scale: 输出坐标的缩放比例，默认1000（0-1000范围）  
      
    Returns:  
        相对坐标列表 [(x1_norm, y1_norm), (x2_norm, y2_norm), ...]  
    """  
    width, height = size[1],size[0]
    relative_points = []  
    for x, y in points:  
        x_norm = int(x * width) 
        y_norm = int(y * height)
        relative_points.append((x_norm, y_norm))  
    return relative_points


def relative_to_absolute_points(points, size) :
    """将图像中的绝对坐标转换为相对坐标
    
    Args:
        points: 绝对坐标列表 [(x1, y1), (x2, y2), ...]
        width: 模型默认的resize图像宽度
        height: 模型默认的resize图像高度
    
    Returns:
        相对坐标列表 [(x1_norm, y1_norm), (x2_norm, y2_norm), ...]
           坐标范围在 [0, 1] 之间
    """

    width, height = size[1],size[0]
    absolute_points = []
    for x, y in points:
        x_norm = x / width
        y_norm = y / height
        absolute_points.append((x_norm, y_norm))
    return absolute_points

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
                x_norm = x
                y_norm = y
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

# 2d BBOX 不同模型配置
MODEL_COORDINATE_CONFIGS = {
    # Qwen系列模型
    "qwen2.5-vl": {
        "is_relative": False, 
        "default_image_size": (-1 ,-1),  
    },
    "qwen3-vl": {
        "is_relative": True,
        "default_image_size": (1000.0, 1000.0),  
    },
    "gemini": {
        "is_relative": True, 
        "default_image_size": (1000.0, 1000.0),  
    }
}