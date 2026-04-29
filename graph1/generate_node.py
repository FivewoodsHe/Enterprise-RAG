from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from graph1.rewrite_node import get_last_human_message
from llm_models.llm_utils import llm_qw3
from utils.log_utils import log


def generate_node(state):
    """
    根据查询问题生成答案

    :param state: 当前状态
    :return: 包含重叙问题的更新后状态
    """
    log.info("---生成答案---")
    messages = state['messages']
    question = get_last_human_message(messages).content

    last_message = messages[-1]
    # RAG检索获取的信息
    docs = last_message.content

    # 提示模板
    prompt = PromptTemplate(
        template="""
          你是一个问答任务助手。请根据以下检索到的上下文内容回答问题。如果不知道答案，请直接说明。回答保持简洁。\n问题：{question} \n上下文：{context} \n回答：  
        """,
        # 明确声明模板中使用的变量名称列表，必须与模板中定义的变量名称一致。调用invoke时，传入的键值必须与变量名称一致。
        input_variables=["question", "context"]
    )

    # 处理链
    rag_chain = prompt | llm_qw3 | StrOutputParser()

    # 执行
    response = rag_chain.invoke({"question": question, "context": docs})
    ai_message = AIMessage(content=response)
    return {"messages": [ai_message]}