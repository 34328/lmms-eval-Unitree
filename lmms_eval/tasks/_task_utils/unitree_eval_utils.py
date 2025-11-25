import re

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





