import asyncio

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

otel_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
tracer_provider = TracerProvider(resource=Resource({"service.name": "autogen-test-agentchat"}))
span_processor = BatchSpanProcessor(otel_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)

# we will get reference this tracer later using its service name
# tracer = trace.get_tracer("autogen-test-agentchat")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient


def search_web_tool(query: str) -> str:
    if "2006-2007" in query:
        return """Here are the total points scored by Miami Heat players in the 2006-2007 season:
        Udonis Haslem: 844 points
        Dwayne Wade: 1397 points
        James Posey: 550 points
        ...
        """
    elif "2007-2008" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2007-2008 is 214."
    elif "2008-2009" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2008-2009 is 398."
    return "No data found."


def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100


async def main() -> None:
    # model_client = OpenAIChatCompletionClient(model="gpt-4o")
    ############################################################
    # 定义模型信息
    ############################################################
    model_info = {
        "name": "deepseek-chat",  # 模型名称，可随意填写
        "parameters": {
            "max_tokens": 2048,  # 每次输出最大token数
            # deepseek官方数据：1个英文字符 ≈ 0.3 个 token。1 个中文字符 ≈ 0.6 个 token。
            "temperature": 0,  # 模型随机性参数，数字越大，生成的结果随机性越大，一般为0.7，
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

    # 创建模型 ==================================================
    model_client = OpenAIChatCompletionClient(model="deepseek-chat",  # 必须与官方给的模型名称一致
                                              base_url="https://api.deepseek.com",  # 调用API地址
                                              api_key="sk-f66788fe1fce45099693f26b56ea89db",
                                              model_info=model_info)

    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            WebSearchAgent: Searches for information
            DataAnalystAgent: Performs calculations

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, summarize the findings and end with "TERMINATE".
        """,
    )

    web_search_agent = AssistantAgent(
        "WebSearchAgent",
        description="An agent for searching information on the web.",
        tools=[search_web_tool],
        model_client=model_client,
        system_message="""
        You are a web search agent.
        Your only tool is search_tool - use it to find information.
        You make only one search call at a time.
        Once you have the results, you never do calculations based on them.
        """,
    )

    data_analyst_agent = AssistantAgent(
        "DataAnalystAgent",
        description="An agent for performing calculations.",
        model_client=model_client,
        tools=[percentage_change_tool],
        system_message="""
        You are a data analyst.
        Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
        If you have not seen the data, ask for it.
        """,
    )

    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination

    selector_prompt = """Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """

    task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"

    tracer = trace.get_tracer("autogen-test-agentchat")
    with tracer.start_as_current_span("runtime"):
        team = SelectorGroupChat(
            [planning_agent, web_search_agent, data_analyst_agent],
            model_client=model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            allow_repeated_speaker=True,
        )
        await Console(team.run_stream(task=task))

    await model_client.close()


asyncio.run(main())
