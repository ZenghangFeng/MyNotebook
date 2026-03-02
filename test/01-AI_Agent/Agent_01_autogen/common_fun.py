import json
from openai import OpenAI


def input_pars(input_str:str):
    """
    可以根据用户输入的文本，从文本中提取出用户想要处理的数据的信息
    :param input_str: 用户输入文本
    :return: 提取的数据信息
    """
    # api_key ==========================================
    api_key = "sk-f66788fe1fce45099693f26b56ea89db"
    # 创建大模型对象 ======================================
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    # 读取提示词模板，以完成信息提取的功能 =====================
    with open(file="prompt_input_parser.txt", encoding="utf-8") as f:
        prompt = f.read()
    # 大模型对话，实现信息提取 ==============================
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': input_str}
        ],
        temperature=0
    )
    res = completion.choices[0].message.content
    res = json.loads(res)
    return res
