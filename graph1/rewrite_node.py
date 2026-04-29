from typing import List

from langchain_core.messages import BaseMessage, HumanMessage

from graph1.graph_state1 import AgentState
from llm_models.llm_utils import llm_qw3
from utils.log_utils import log


def get_last_human_message(messages: List[BaseMessage]) -> HumanMessage:
    # 反向遍历信息列表，找到最后一个人类消息
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    raise ValueError("No human message found in this list!!")



def rewrite_node(state: AgentState):
    """
    转换查询以生成更好的问题。

    :param state: 当前状态
    :return:  dict: 包含重述问题的更新后状态
    """
    log.info("---查询转换优化---")
    messages = state['messages']
    question = get_last_human_message(messages).content

    msg: list[HumanMessage] = [HumanMessage(content=f"""
    分析输入的问题并尝试理解潜在的语义意图/含义。\n
    这是初始问题：{question}\n
    尝试生成一个更好的问题，并返回该问题。
    """)]

    # 获取模型生成的问题
    new_question = llm_qw3.invoke(msg)
    return {"messages" : [new_question]}
