# 配置模型信息
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# https://zhuanlan.zhihu.com/p/27859627155

model_info = {
    "name": "deepseek-chat", # 模型名称，可随意填写
    "parameters": {
        "max_tokens": 2048,  # 每次输出最大token数
                             # deepseek官方数据：1个英文字符 ≈ 0.3 个 token。1 个中文字符 ≈ 0.6 个 token。
        "temperature": 0.4,  # 模型随机性参数，数字越大，生成的结果随机性越大，一般为0.7，
                             # 如果希望AI提供更多的想法，可以调大该数字
        "top_p": 0.9,  # 模型随机性参数，接近 1 时：模型几乎会考虑所有可能的词，只有概率极低的词才会被排除，随机性也越强；
                       # 接近 0 时：只有概率非常高的极少数词会被考虑，这会使模型的输出变得非常保守和确定
    },
    "family": "gpt-4o",  # 必填字段，model属于的类别
    "functions": [],  # 非必填字段，如果模型支持函数调用，可以在这里定义函数信息
    "vision": False,  # 必填字段，模型是否支持图像输入
    "json_output": True,  # 必填字段，模型是否支持json格式输出
    "function_calling": True  # 必填字段，模型是否支持函数调用，如果模型需要使用工具函数，该字段为true
}

# 创建模型
model_client = OpenAIChatCompletionClient(model="deepseek-chat", # 必须与官方给的模型名称一致
                                          base_url="https://api.deepseek.com", # 调用API地址
                                          api_key="sk-f66788fe1fce45099693f26b56ea89db",
                                          model_info=model_info)

# 模型调用的工具函数
async def get_weather(city: str) -> str:
    """获取城市天气信息 这里只是一个demo 直接输出了一句话"""
    return f"{city}的天气为晴天，温度为23摄氏度。"

# 定义智能体
agent = AssistantAgent(
    name="weather_agent", # 智能体名称 建议以工作内容命名
    model_client=model_client, # 智能体使用的模型
    tools=[get_weather], # 智能体可以调用的工具
    system_message="你是一个有用的助手", # 智能体描述 相当于写一段提示词预训练模型 demo功能少所以写的简单
    reflect_on_tool_use=True, # 调用工具的结果还需要经过模型推理
    model_client_stream=True,  # 允许模型流式输出
)

async def main() -> None:
    await Console(agent.run_stream(task="纽约今天天气怎么样？"))

# await main() # 直接在python中使用demo会在这里报错 不知道是在什么场景下这么用

# 直接运行python脚本使用这段代码
if __name__ == "__main__":
    asyncio.run(main())