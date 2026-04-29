import uuid
from typing import Literal

from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from graph1.agent_node import agent_node
from graph1.generate_node import generate_node
from graph1.graph_state1 import Grade, AgentState
from graph1.rewrite_node import rewrite_node, get_last_human_message
from llm_models.llm_utils import llm_qw3, llm_ds
from tools.draw_png import draw_graph
from tools.retriever_tools import retriever_tool
from utils.log_utils import log
from utils.print_utils import _print_event


def grade_documents(state) -> Literal["generate", "rewrite"]: # Literal类型注解工具，严格表明返回值
    """
    判断检索到的文档是否与问题相关。
    参数:
        state (messages): 当前状态
    返回:
        str: 判断结果，文档是否相关
    """
    log.info("---检查document的相关性---")
    #  带结构化输出的LLM
    llm_with_structured = llm_qw3.with_structured_output(Grade)

    # faq: 提示模板， 如果 {"score"} 是模板中应作为静态文本显示的部分，请修改模板定义，使用双大括号转义：
    prompt = PromptTemplate(
        template="""你是一个评估检索文档与用户问题相关性的评分器。\n
            这是检索到的文档：\n\n {context} \n\n
            这是用户的问题：{question} \n
            如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n
            给出二元评分 'yes' 或 'no' 来表示文档是否与问题相关。\
            请严格按照以下JSON格式返回结果：
            {{\"score\":\"yes或no\"}}
            """,
        input_variables=["context", "question"],
    )

    # 处理链
    chain = prompt | llm_with_structured

    messages = state["messages"]
    last_message = messages[-1]

    question = get_last_human_message(messages).content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.score

    if score == "yes":
        print("---输出：文档相关---")
        return "generate"

    else:
        print("---输出：文档不相关---")
        print(score)
        return "rewrite"



# 定义一个新的状态图
workflow = StateGraph(AgentState)
# 添加节点
workflow.add_node('agent', agent_node)
workflow.add_node('retrieve', ToolNode([retriever_tool])) # RAG检索节点
workflow.add_node('rewrite', rewrite_node)
workflow.add_node('generate', generate_node)

# 添加边和条件边
workflow.add_edge(START, 'agent')
# 判断用户问题是否进行RAG检索的条件边 todo:tools_condition看一下这个是什么逻辑
workflow.add_conditional_edges(
    source='agent',
    path=tools_condition, # 这是默认的工具条件边
    path_map={
        'tools': 'retrieve',
        END: END
    }
)

# 相关性评价的条件边。路由映射如果一致，则path_map可以不写。
workflow.add_conditional_edges(
    source = 'retrieve',
    path=grade_documents,
    # 返回值和路由映射一致，则path_map可以不写。
    path_map={
        'rewrite':'rewrite',
        'generate':'generate'
    }
)

workflow.add_edge('rewrite', 'agent')
workflow.add_edge('generate', END)


memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# 绘制图，需要翻墙
# draw_graph(graph, 'graph_rag1.png')

config = {
    "configurable":{
        "thread_id": str(uuid.uuid4())
    }
}

_printed = set()

# 执行工作流
while True:
    question = input('请输入问题：')
    if question.lower() in ['q', 'exit', 'quit']:
        log.info('对话结束，拜拜！')
        break
    else:
        inputs = {
            "messages" : [('user', question)]
        }
        events = graph.stream(inputs, config=config, stream_mode='values')
        # 打印消息
        for event in events:
            _print_event(event, _printed)















