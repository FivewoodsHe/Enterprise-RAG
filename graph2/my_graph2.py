from pprint import pprint

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graph2.chain_of_answer_grader import answer_grader_chain
from graph2.chain_of_grade_hallucinations import hallucination_grader_chain
from graph2.graph_state2 import GraphState
from graph2.node_generate import generate
from graph2.node_grade_documents import grade_documents
from graph2.node_retrieve import retrieve
from graph2.node_transform_query import transform_query
from graph2.node_web_search import web_search
from graph2.chain_of_question_route import question_router_chain
from tools.draw_png import draw_graph
from utils.log_utils import log


def route_question(state: GraphState):
    """
    将问题路由到网络搜索或RAG流程
    :param state:  当前图状态，包含用户问题
    :return: 下一个节点的名称
    """
    log.info("--- 问题路由 ---")
    question = state['question']
    source = question_router_chain.invoke({"question": question})  # 调用问题路由处理链


    # # todo 增加返回值校验
    # if not hasattr(source, 'datasource'):
    #     log.error("路由链未返回有效的数据源信息")
    #     return None

    # 根据路由结果确定下一个节点
    if source.datasource == 'web_search':
        return 'web_search'
    elif source.datasource == 'vectorstore':
        return 'retrieve'
    else:
        log.error("无效的源")
        return None

def decide_to_generate(state):
    """

    :return:
    """
    log.info("----文档评估后走向----")  # 阶段标识
    filtered_documents = state['documents'] # 获取已过滤的文档
    transform_count = state.get("transform_count", 0)

    if not filtered_documents:
        if transform_count >= 2:
            log.info("----决策：所有文档与问题无关且循环2次，转为web查询问题----")
            return 'web_search'
        log.info("----决策：所有文档与问题无关，转为问题优化节点----")
        return "transform_query"
    else:
        log.info("----文档与问题有关，转为回答生成节点----")
        return 'generate'


def grade_generation_v_documents_and_question(state):
    """
    评估生成的结果是否基于文档并正确回答问题
    :param state: (dict) 当前图状态，包含问题、文档和生成结果
    :return: 下一个节点的名称（userful、not userful、 not supported)
    """
    log.info("----评估生成的结果是否基于文档并正确回答问题----")
    question = state['question'] # 用户问题
    documents = state['documents'] # 获取参考文档
    generation = state['generation'] # 获取生成的结果

    # 检查生成是否基于文档
    score = hallucination_grader_chain.invoke({"text":documents, "generation" : generation})
    grade = score.binary_score

    if grade == "yes": # 基于文档生成
        log.info("----判定：生成的内容基于文档----")
        # 评估：生成回答与问题的匹配度
        score =  answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes": # 匹配度高
            log.info("----判定：生成的内容基于文档且与问题匹配，-》 END节点----")
            return "userful"
        else:
            log.info("----判定：生成的内容基于文档但与问题不匹配，-》优化问题节点 ----")
            return "not userful" # 返回无用结果
    else:
        log.info("----判定：生成的内容不基于文档，将重新尝试生成答案----")
        return "not supported"


# 具有自我纠正的工作流图
workflow = StateGraph(GraphState)

# 定义状态节点
workflow.add_node('web_search', web_search)  # 网络搜索节点
workflow.add_node('retrieve', retrieve)  # 文档检索节点
workflow.add_node('grade_documents', grade_documents)  # 文档相关性评分节点
workflow.add_node('generate', generate)  # 回答生成节点
workflow.add_node('transform_query', transform_query)  # 查询优化节点

# 起始路由判断
workflow.add_conditional_edges(
    START,
    route_question,
    path_map={
        "web_search": "web_search",
        "retrieve": "retrieve"
    }
)

# 添加固定边
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")



# 文档评估后的条件分支（1.循环次数2次以内、文档无关
workflow.add_conditional_edges(
    'grade_documents',
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        'web_search':'web_search',
    }
)

# 生成回答后的条件分支
workflow.add_conditional_edges(
    source='generate',
    path=grade_generation_v_documents_and_question,
    path_map={
        "not supported":"generate",
        "userful":END,
        "not userful":"transform_query",
    }
)

workflow.add_edge("transform_query", "retrieve")


# 编译工作流
graph = workflow.compile()

# 绘制流程图
draw_graph(graph, "graph2.png")


# if __name__ == '__main__':
#
#     # 执行工作流
#     _printed = set()  # set集合，避免重复打印
#
#     while True:
#         question = input('用户：')
#         if question.lower() in ['q', 'exit', 'quit']:
#             print('对话结束，拜拜！')
#             break
#         else:
#             inputs = {
#                 "question": question
#             }
#             # 流式执行工作流
#             for output in graph.stream(inputs):
#                 for key, value in output.items():
#                     # 打印当前节点名称
#                     pprint(f"Node '{key}':")  # 显示当前执行的节点名称
#                     # 可选：打印每个节点的完整状态信息
#                     # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
#                 pprint("\n---\n")  # 节点分隔线
#
#             # 打印最终生成结果
#             pprint(value["generation"])  # 输出最终生成的回答内容
























