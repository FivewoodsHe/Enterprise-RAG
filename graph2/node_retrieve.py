from graph2.graph_state2 import GraphState
from tools.retriever_tools import retriever
from utils.log_utils import log


def retrieve(state: GraphState):
    """
        向量库中检索相关文档
    :param state:(dict) 当前图状态，
    :return: state(dict) 更新后的状态，新增包含检索结果的documents字段
    """
    log.info('--- 知识库中检索 ---')

    question = state['question']
    # 文档检索
    documents = retriever.invoke(question) # 调用检索器获取相关文档
    return {"question": question, "documents": documents}


