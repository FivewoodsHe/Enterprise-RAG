from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from graph2.graph_state2 import GraphState
from llm_models.llm_utils import llm_qw3
from utils.log_utils import log


def transform_query(state:GraphState):
    """
    优化用户的问题，生成更合适的查询语句

    :param state: (dict) 当前图状态，包含用户问题和检索结果
    :return: (dict) 优化后的状态，question替换为优化后的问题
    """
    log.info("--- 问题优化 ---") # 打印阶段标识
    question = state['question']
    documents = state['documents']
    transform_count = state.get("transform_count", 0)

    # 提示词模板优化   -问题重写优化
    system = """
        作为问题重写器，您需要将输入问题转换为更适合向量数据库检索的优化版本。\n
         请分析输入问题并理解其背后的语义意图/真实含义。
    """
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        # faq: 在langchain框架中，支持多个角色标识符，human和user等价，"ai" ≡ "assistant"
        ("human", "问题: {question}\n 请生成一个优化后的问题。")
    ])

    # 构建问题重写处理链
    question_rewrite_chain = (
            re_write_prompt | llm_qw3
            | StrOutputParser() # 将输出解析为字符串
    )

    # 问题重写
    better_question = question_rewrite_chain.invoke({"question":question})

    return {"question": better_question, "documents": documents, "transform_count": transform_count + 1}
