from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

"""
    定义了图中节点之间message传递的状态
"""
class AgentState(TypedDict):
    # 用于存储消息列表并定义如何处理状态更新。add_messages 表示追加信息，该函数定义如何处理状态更新
    messages: Annotated[list[BaseMessage], add_messages]

# 大模型返回结果需要转换的数据模型
class Grade(BaseModel):
    """相关性检查的二元评分"""
    score: str = Field(description="二元相关性评分，值为 'yes' 或 'no' ")