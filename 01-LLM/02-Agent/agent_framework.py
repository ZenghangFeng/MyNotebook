import json
import re
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import openai
from openai import OpenAI


# ============ 基础数据类型和枚举 ============
class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"


class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    TOOL_RESULT = "tool_result"


@dataclass
class Message:
    content: str
    role: str  # OpenAI消息格式中的role
    msg_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI API要求的消息格式[citation:1][citation:4]"""
        return {"role": self.role, "content": self.content}


# ============ 记忆系统 ============
class Memory(ABC):
    @abstractmethod
    def add(self, message: Message) -> None:
        pass

    @abstractmethod
    def get_conversation_history(self, max_tokens: int = 4000) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class ConversationMemory(Memory):
    """基于对话历史的记忆系统，适配OpenAI API[citation:4][citation:7]"""

    def __init__(self, max_messages: int = 50):
        self.messages: List[Message] = []
        self.max_messages = max_messages

    def add(self, message: Message) -> None:
        self.messages.append(message)
        # 控制记忆长度，避免超过模型上下文限制[citation:4]
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_conversation_history(self, max_tokens: int = 4000) -> List[Dict[str, Any]]:
        """获取适合OpenAI API的对话历史[citation:7]"""
        # 简化实现：返回最近的对话
        openai_messages = []
        token_count = 0

        for msg in reversed(self.messages):
            # 估算token数（简化：假设中文字符2个token，英文1个）
            msg_tokens = len(msg.content) * 2 if any('\u4e00' <= c <= '\u9fff' for c in msg.content) else len(
                msg.content)
            if token_count + msg_tokens > max_tokens:
                break
            openai_messages.insert(0, msg.to_openai_format())
            token_count += msg_tokens

        return openai_messages

    def clear(self) -> None:
        self.messages.clear()


# ============ 工具系统 ============
@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI函数调用格式[citation:2]"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters.get("properties", {}),
                    "required": self.parameters.get("required", []),
                    "additionalProperties": self.parameters.get("additionalProperties", False)
                },
                "strict": True  # 启用结构化输出[citation:2]
            }
        }

    def execute(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """获取OpenAI API可用的工具列表[citation:2]"""
        return [tool.to_openai_format() for tool in self.tools.values()]


# ============ 预定义工具（带完整参数定义） ============
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全评估表达式
        expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def search_web(query: str, max_results: int = 3) -> str:
    """搜索网络信息[citation:9]

    Args:
        query: 搜索查询词
        max_results: 最大结果数量
    """
    # 实际应用中会调用真实的搜索引擎API
    # 这里模拟返回结果[citation:9]
    return f"搜索 '{query}' 的结果:\n1. 结果1相关信息\n2. 结果2相关信息\n3. 结果3相关信息"


def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间

    Args:
        timezone: 时区，默认为亚洲/上海
    """
    from datetime import datetime
    import pytz

    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return f"当前时间（{timezone}）: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"
    except:
        current_time = datetime.now()
        return f"当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"


def save_note(content: str, title: str = "未命名笔记") -> str:
    """保存笔记到文件

    Args:
        content: 笔记内容
        title: 笔记标题
    """
    import os
    os.makedirs("notes", exist_ok=True)
    filename = f"notes/{title.replace(' ', '_')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"笔记已保存到 {filename}"


# ============ OpenAI LLM 接口 ============
class OpenAIClient:
    """OpenAI API客户端封装[citation:1][citation:4]"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = 0.7
        self.max_tokens = 1000

    def chat_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[
        str, Any]:
        """调用OpenAI聊天完成API[citation:1][citation:7]"""
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"  # 让模型决定是否调用工具[citation:2]

            response = self.client.chat.completions.create(**params)
            return response
        except Exception as e:
            print(f"OpenAI API调用错误: {str(e)}")
            raise


# ============ 核心Agent（重构版） ============
class OpenAIAgent:
    """基于OpenAI大模型的智能Agent"""

    def __init__(self, name: str = "智能助手", model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.name = name
        self.status = AgentStatus.IDLE

        # 初始化OpenAI客户端[citation:1]
        self.llm = OpenAIClient(api_key=api_key, model=model)

        # 初始化组件
        self.memory = ConversationMemory(max_messages=30)
        self.tool_registry = ToolRegistry()

        # 注册工具
        self._register_tools()

        # 系统提示词
        self.system_prompt = f"""你是一个名为{name}的AI助手。你可以使用工具来帮助用户解决问题。

你可以使用的工具：
{self._get_tools_description()}

对话规则：
1. 当用户需要计算、搜索、获取时间或保存笔记时，使用相应的工具
2. 保持对话友好和专业
3. 如果工具调用失败，向用户解释并尝试其他方法
4. 尽量在一次对话中解决用户的问题
"""

    def _register_tools(self):
        """注册所有可用的工具"""
        tools = [
            Tool(
                name="calculator",
                description="计算数学表达式",
                func=calculator,
                parameters={
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "要计算的数学表达式，如 '125 + 37 * 2'"
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": False
                }
            ),
            Tool(
                name="search_web",
                description="搜索最新的网络信息",
                func=search_web,
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询关键词"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "最大结果数量，默认为3",
                            "default": 3
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            ),
            Tool(
                name="get_current_time",
                description="获取当前时间",
                func=get_current_time,
                parameters={
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "时区，如 'Asia/Shanghai'、'America/New_York'",
                            "default": "Asia/Shanghai"
                        }
                    },
                    "required": [],
                    "additionalProperties": False
                }
            ),
            Tool(
                name="save_note",
                description="保存笔记到文件",
                func=save_note,
                parameters={
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "笔记内容"
                        },
                        "title": {
                            "type": "string",
                            "description": "笔记标题，默认为'未命名笔记'",
                            "default": "未命名笔记"
                        }
                    },
                    "required": ["content"],
                    "additionalProperties": False
                }
            )
        ]

        for tool in tools:
            self.tool_registry.register(tool)

    def _get_tools_description(self) -> str:
        """获取工具描述字符串"""
        desc = []
        for name, tool in self.tool_registry.tools.items():
            desc.append(f"- {name}: {tool.description}")
        return "\n".join(desc)

    def process(self, user_input: str, max_iterations: int = 5) -> str:
        """处理用户输入的主要方法"""
        self.status = AgentStatus.THINKING

        # 1. 保存用户消息
        user_msg = Message(
            content=user_input,
            role="user",
            msg_type=MessageType.USER
        )
        self.memory.add(user_msg)

        # 2. 构建对话历史（包含系统提示）
        conversation = [{"role": "system", "content": self.system_prompt}]
        conversation.extend(self.memory.get_conversation_history())

        # 3. 准备工具列表[citation:2]
        tools = self.tool_registry.get_openai_tools()

        iteration = 0
        final_response = None

        while iteration < max_iterations:
            iteration += 1

            # 4. 调用OpenAI API[citation:1][citation:2]
            response = self.llm.chat_completion(conversation, tools if iteration == 1 else None)
            message = response.choices[0].message

            # 5. 检查是否有工具调用[citation:2]
            if hasattr(message, 'tool_calls') and message.tool_calls:
                self.status = AgentStatus.ACTING

                # 处理每个工具调用
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    # 执行工具
                    tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        try:
                            tool_result = tool.execute(**tool_args)
                        except Exception as e:
                            tool_result = f"工具执行错误: {str(e)}"
                    else:
                        tool_result = f"未知工具: {tool_name}"

                    # 保存工具调用和结果到对话历史
                    tool_msg = Message(
                        content=json.dumps(tool_args, ensure_ascii=False),
                        role="assistant",
                        msg_type=MessageType.TOOL
                    )
                    self.memory.add(tool_msg)

                    result_msg = Message(
                        content=tool_result,
                        role="tool",
                        msg_type=MessageType.TOOL_RESULT
                    )
                    self.memory.add(result_msg)

                    # 添加到对话上下文，用于下一轮
                    conversation.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_call.function.arguments
                            }
                        }]
                    })
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # 继续下一轮对话（让模型基于工具结果生成回复）
                continue

            else:
                # 没有工具调用，生成最终回复
                final_response = message.content

                # 保存助手回复到记忆
                assistant_msg = Message(
                    content=final_response,
                    role="assistant",
                    msg_type=MessageType.ASSISTANT
                )
                self.memory.add(assistant_msg)

                self.status = AgentStatus.IDLE
                break

        if not final_response:
            final_response = "抱歉，我无法处理您的请求。请稍后再试或尝试更简单的查询。"

        return final_response

    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态信息"""
        return {
            "name": self.name,
            "status": self.status.value,
            "model": self.llm.model,
            "memory_size": len(self.memory.messages),
            "available_tools": list(self.tool_registry.tools.keys())
        }

    def reset(self) -> None:
        """重置对话"""
        self.memory.clear()
        self.status = AgentStatus.IDLE


# ============ 命令行界面 ============
class AgentCLI:
    def __init__(self, agent: OpenAIAgent):
        self.agent = agent

    def run(self):
        """运行交互式命令行界面"""
        print(f"=== {self.agent.name} 启动 ===")
        print(f"模型: {self.agent.llm.model}")
        print("输入 '退出' 或 'quit' 结束对话")
        print("输入 '状态' 查看Agent状态")
        print("输入 '重置' 清除对话历史")
        print("=" * 40)

        while True:
            try:
                user_input = input("\n您: ").strip()

                if user_input.lower() in ['退出', 'quit', 'exit']:
                    print(f"\n{self.agent.name}: 再见！")
                    break

                elif user_input == '状态':
                    status = self.agent.get_status()
                    print(f"Agent状态:")
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                    continue

                elif user_input == '重置':
                    self.agent.reset()
                    print("对话历史已清除")
                    continue

                # 处理用户输入
                print(f"{self.agent.name}: ", end="", flush=True)
                response = self.agent.process(user_input)
                print(response)

            except KeyboardInterrupt:
                print(f"\n{self.agent.name}: 再见！")
                break
            except Exception as e:
                print(f"错误: {str(e)}")


# ============ 使用示例 ============
def main():
    """主函数：演示OpenAI Agent的使用"""

    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("请设置OPENAI_API_KEY环境变量")
        print("例如: export OPENAI_API_KEY='你的密钥'")
        return

    # 创建Agent实例（可以切换为gpt-4、gpt-4o等模型[citation:10]）
    agent = OpenAIAgent(
        name="智囊AI",
        model="gpt-3.5-turbo",  # 可以改为"gpt-4"、"gpt-4o"等[citation:10]
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 创建并运行命令行界面
    cli = AgentCLI(agent)
    cli.run()


def test_agent():
    """测试Agent功能"""
    print("=== 测试OpenAI Agent框架 ===\n")

    # 注意：需要先设置OPENAI_API_KEY环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("跳过测试：需要设置OPENAI_API_KEY环境变量")
        return

    # 创建测试Agent
    agent = OpenAIAgent(
        name="测试助手",
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 测试用例
    test_cases = [
        "计算一下45加上67乘以2等于多少？",
        "现在几点了？",
        "搜索一下人工智能的最新发展",
        "帮我保存一个笔记，标题是'会议记录'，内容是'明天上午10点开会讨论项目进度'"
    ]

    for test_input in test_cases:
        print(f"测试输入: {test_input}")
        try:
            response = agent.process(test_input)
            print(f"Agent回复: {response[:150]}...\n")
            print("-" * 60)
        except Exception as e:
            print(f"错误: {str(e)}\n")

        # 重置对话
        agent.reset()


# ============ 高级功能：流式输出示例 ============
class StreamingAgent(OpenAIAgent):
    """支持流式输出的Agent[citation:5]"""

    def stream_process(self, user_input: str, callback: Callable[[str], None]):
        """流式处理用户输入[citation:5]"""
        self.status = AgentStatus.THINKING

        # 保存用户消息
        user_msg = Message(
            content=user_input,
            role="user",
            msg_type=MessageType.USER
        )
        self.memory.add(user_msg)

        # 构建对话历史
        conversation = [{"role": "system", "content": self.system_prompt}]
        conversation.extend(self.memory.get_conversation_history())

        # 流式调用OpenAI API[citation:5]
        try:
            stream = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=conversation,
                stream=True,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    callback(content)

            # 保存助手回复到记忆
            assistant_msg = Message(
                content=full_response,
                role="assistant",
                msg_type=MessageType.ASSISTANT
            )
            self.memory.add(assistant_msg)

            self.status = AgentStatus.IDLE
            return full_response

        except Exception as e:
            error_msg = f"流式处理错误: {str(e)}"
            callback(error_msg)
            return error_msg


if __name__ == "__main__":
    # 运行测试
    # test_agent()

    # 运行主程序（交互式界面）
    main()