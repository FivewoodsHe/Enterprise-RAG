from langchain_core.documents import Document

from graph2.graph_state2 import GraphState
from llm_models.llm_utils import web_search_tool
from utils.log_utils import log


def web_search(state: GraphState):
    """
    进行网络搜索

    :param state: (dict) 当前图的状态，包含优化后的问题
    :return: state(dict) 更新后的装填， documents替换为网络搜索的结果
    """
    log.info("---web search---")
    question = state['question']

    # 执行网络搜索
    docs = web_search_tool.invoke({'query': question})

    # faq: 该错误表示代码试图将字符串与类型对象进行拼接操作，Python不允许直接连接str和type类型
    # print('******web results: ' + type(docs))
    print(docs)

    web_results = '\n'.join([d['content'] for d in docs]) # 合并搜索结果
    web_results = Document(page_content=web_results) # 转换为文档格式

    return {"question": question, "documents":web_results}
