from langgraph.graph.state import CompiledStateGraph

from utils.log_utils import log

# 绘制agent的节点图
def draw_graph(graph: CompiledStateGraph, file_name:str):
    try:
        mermaid_code = graph.get_graph().draw_mermaid_png()
        with open(file_name, 'wb') as f:
            f.write(mermaid_code)
    except Exception as e:
        log.exception(e)