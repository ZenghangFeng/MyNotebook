# 配置模型信息
import asyncio

import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from matplotlib import pyplot as plt
from openai import OpenAI
from common_fun import input_pars

"""
给定一些可能会被用于分析的数据，用户输入一段文本描述分析需求，
Agent首先从用户输入文本中提取信息，确定数据范围，
再进行分析并返回分析结果
"""

############################################################
# 定义模型信息
############################################################
model_info = {
    "name": "deepseek-chat",            # 模型名称，可随意填写
    "parameters": {
        "max_tokens": 2048,             # 每次输出最大token数
                                        # deepseek官方数据：1个英文字符 ≈ 0.3 个 token。1 个中文字符 ≈ 0.6 个 token。
        "temperature": 0,               # 模型随机性参数，数字越大，生成的结果随机性越大，一般为0.7，
                                        # 如果希望AI提供更多的想法，可以调大该数字
        "top_p": 0.9,                   # 模型随机性参数，接近 1 时：模型几乎会考虑所有可能的词，只有概率极低的词才会被排除，随机性也越强；
                                        # 接近 0 时：只有概率非常高的极少数词会被考虑，这会使模型的输出变得非常保守和确定
    },
    "family": "gpt-4o",                 # 必填字段，model属于的类别
    "functions": [],                    # 非必填字段，如果模型支持函数调用，可以在这里定义函数信息
    "vision": False,                    # 必填字段，模型是否支持图像输入
    "json_output": True,                # 必填字段，模型是否支持json格式输出
    "function_calling": True            # 必填字段，模型是否支持函数调用，如果模型需要使用工具函数，该字段为true
}

# 创建模型 ==================================================
model_client = OpenAIChatCompletionClient(model="deepseek-chat",                    # 必须与官方给的模型名称一致
                                          base_url="https://api.deepseek.com",      # 调用API地址
                                          api_key="sk-f66788fe1fce45099693f26b56ea89db",
                                          model_info=model_info)


############################################################
# 模型调用的工具函数
############################################################
async def plot_scatter(user_str:str):
    """
    读取用户输入的文本，确定excel文件，根据表格中选定的两列数据绘图
    :param user_str: 用户输入的文本
    :return:
    """

    try:
        dit_data = input_pars(user_str)
        file_name = dit_data["文件名称"]
        cols = dit_data["列名称"].split("/")
        df = pd.read_excel(file_name+".xlsx")
        x = df[cols[0]]
        y = df[cols[1]]
        plt.rcParams["font.sans-serif"] = ["SimHei"]                   # 设置显示中文字体
        plt.scatter(x, y, color='r', label="")                         # 绘制散点图
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(file_name.split(".")[0] + ".png", transparent=True)
        print("------------------绘图完成-------------------")
        res = "正常完成绘图"

    except ValueError as E:
        print("-----------------文件名称错误-----------------")
        res = "文件名称错误"

    return res


async def pic_analysis(file_name:str):
    """
    读取输入的图像，分析图像上的点的分布，判断x轴和y轴变量的相关性
    :param file_name:
    :return:
    """
    # 千问 API ==================================================
    api_key_qw = "sk-98b706212bda4bfea375068e0558c2bc"
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key_qw,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    #  Base64 编码格式图片加载 =====================================
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # 对话解析图片 ================================================
    try:
        base64_image = encode_image(file_name)
        user_text = "从图中点的分布情况判断x轴和y轴变量的相关性"

        completion = client.chat.completions.create(
            model="qwen-omni-turbo",
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            # 设置输出数据的模态，当前支持["text"]
            modalities=["text"],
            # stream 必须设置为 True，否则会报错
            stream=True,
            stream_options={"include_usage": True},
        )

        # 整合图片分析的结果 =============================================
        res = ""
        for chunk in completion:
            if chunk.choices:
                tmp = chunk.choices[0].delta.content
                if tmp:
                    res += tmp

        print("---------------------图像分析完成-------------------")

    except ValueError as E:
        res = "工具调用失败，无返回结果"
        print("----------------------图像不存在--------------------")

    return res


############################################################
# 定义智能体
############################################################
system_message="你是一个有用的助手，可以对用户提供的数据文件进行分析，可以进行的分析包括：1）根据用户选择的两列数据绘制图像；2）分析绘制的图片并输出结论"
agent = AssistantAgent(
    name="data_analysis_agent",             # 智能体名称 建议以工作内容命名
    model_client=model_client,              # 智能体使用的模型
    tools=[plot_scatter, pic_analysis],     # 智能体可以调用的工具
    system_message=system_message,          # 智能体描述 相当于写一段提示词预训练模型 demo功能少所以写的简单
    reflect_on_tool_use=True,               # 调用工具的结果还需要经过模型推理
    model_client_stream=True,               # 允许模型流式输出
)

"""
# 单轮对话 ===================================================
async def user_interaction(task) -> None:
    await Console(agent.run_stream(task=task))


# 直接运行python脚本使用这段代码
if __name__ == "__main__":
    task = "请绘制 {A20250401} 这个Excel里面的 {总收入} 和 {总消费} 两列数据的散点图"
    asyncio.run(user_interaction(task = task))
"""

# https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/human-in-the-loop.html
# 多轮对话的版本 ==============================================
async def user_interaction(task) -> None:
    team = RoundRobinGroupChat([agent], max_turns=1)
    while True:
        # Run the conversation and stream to the console.
        stream = team.run_stream(task=task)
        # Use asyncio.run(...) when running in a script.
        await Console(stream)
        # Get the user response.
        task = input("Enter your feedback (type 'exit' to leave): ")
        if task.lower().strip() == "exit":
            break

if __name__ == "__main__":
    task = "请绘制 {A20250401} 这个Excel里面的 {总收入} 和 {总消费} 两列数据的散点图"
    asyncio.run(user_interaction(task = task))