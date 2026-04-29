from graph1.graph_state1 import AgentState
from llm_models.llm_utils import llm_qw3
from tools.retriever_tools import retriever_tool
from utils.log_utils import log


# agent节点
def agent_node(state: AgentState):
    """
    调用智能体模型基于当前状态生成响应。根据问题，它会决定使用检索工具检索，或者直接结束。
    :param state: 当前状态
    :return: 更新后的状态，包含附加到消息中的智能体响应
    """
    log.info("---进入agent节点---")
    messages = state['messages']

    model = llm_qw3.bind_tools([retriever_tool])
    response = model.invoke([messages[-1]])

    return {'messages': [response]}
