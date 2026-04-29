from typing import TypedDict, List

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    表示图处理流程的状态信息

    参数:
        question: 用户提出的问题文本
        generation: 语言模型生成的回答文本
        transform_count: 传换查询的次数
        documents: 检索到的相关文档列表
    """
    question: str # 存储当前用户的问题
    transform_count: int # 转换查询的次数
    generation: str # 存储LLM生成的内容
    documents: List[Document]