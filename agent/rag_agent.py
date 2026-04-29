from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from llm_models.llm_utils import llm_qw3
from tools.retriever_tools import retriever_tool

import sys
import traceback


# 添加自定义异常钩子
def custom_excepthook(exc_type, exc_value, exc_traceback):
    print(f"异常类型: {exc_type}")
    print(f"异常值: {exc_value}")
    traceback.print_tb(exc_traceback)

sys.excepthook = custom_excepthook

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个智能助手，尽可能通过调用工具回答用户的问题。'),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ('human', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad', optional=True)
])

agent = create_tool_calling_agent(
    llm=llm_qw3,
    tools=[retriever_tool],
    prompt=prompt,
)

# 这里使用langchain中AgentExecutor，后续版本agent都放在langgraph中
executor = AgentExecutor(agent=agent, tools=[retriever_tool])

def query1():
    resp1 = executor.invoke(input={'input': "EUV光源功率相关信息？"})
    print(resp1)


store = {} # 使用内存：存储历史信息的集合
def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def query2():
    # 在executor外包上包一层，添加历史记录
    agent_with_history = RunnableWithMessageHistory(
        runnable=executor,
        get_session_history=get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history'
    )

    resp2 = agent_with_history.invoke(input={'input': "最先进的纳米清洗技术有哪些？"}, config={'configurable': {'session_id': 'sz123'}})
    print(resp2)

    resp3 = agent_with_history.invoke(input={'input': "芯片封装技术有哪些？"}, config={'configurable': {'session_id': 'sz123'}})
    print(resp3)



if __name__ == '__main__':
    # query1() 不带历史记录的请求

    query2()