from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from graph2.graph_state2 import GraphState
from llm_models.llm_utils import llm_qw3
from utils.log_utils import log


def generate(state: GraphState):
    """
    生成回答

    :param state:  当前图状态，包含问题和检索结果
    :return: 更新后的状态，新增包含结果的generation字段
    """
    log.info("----生成答案----")

    question = state['question']
    documents = state['documents']

    prompt = PromptTemplate(
        template="""
        你是一个问答任务助手。请根据以下检索到的上下文内容回答问题。
        如果不知道答案，请直接说明。回答尽量详细。\n问题：{question} \n上下文：{context} \n回答： 
        """,
        input_variables=["question", "context"]
    )

    # 构建问题生成的处理链
    rag_chain = prompt | llm_qw3 | StrOutputParser()

    # 格式化检索到的文档
    def format_docs(docs):
        """将多个文档合并为一个字符串，用两个换行符分隔每个文档"""
        if isinstance(docs, list):
            return "\n\n".join(doc.page_content for doc in docs)
        else:
            return "\n\n" + docs.page_content

    # 回答生成过程
    generation = rag_chain.invoke({"question": question, "context": format_docs(documents)})
    # todo 这里是否可以不返回documents
    return {"documents": documents, "question": question, "generation": generation}
