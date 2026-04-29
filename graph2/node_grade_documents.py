from graph2.chain_of_grader import retrieval_grader_chain
from graph2.graph_state2 import GraphState
from utils.log_utils import log


def grade_documents(state: GraphState):
    """
    评估检索到的文档与问题的相关性

    :param state: (dict) 当前图的状态，包含问题和检索结果
    :return: (dict) 更新后的状态，documents字段仅保留相关文档
    """
    log.info("--- 文档相关性评分 ---")

    question = state['question']
    documents = state['documents']

    # 文档评分与过滤
    filtered_docs = [] # 初始化相关文档列表
    for d in documents: # 遍历文档
        score = retrieval_grader_chain.invoke( # 调用评分器评估文档相关性
            {"question": question, "document": d.page_content}
        )

        grade = score.binary_score
        if grade == "yes":
            log.info("----grade_documents：与文档相关----")
            filtered_docs.append(d)
        else:
            log.info("----grade_documents：与文档不相关----")
            weather_server_config = {
                "url": "http://127.0.0.1:8008/sse",
                'transport':'sse'
            }

    return {"question": question, "documents": filtered_docs}








