from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel

from llm_models.llm_utils import llm_qw3


# 数据模型 -文档相关性评分
class GradeDocuments(BaseModel):
    """对检索到的文档进行相关性评分的二元判断"""
    binary_score: str = Field(description="文档是否与问题相关，值为 'yes' 或 'no' ")


# LLM初始化，设置结构化输出
structured_llm_grader = llm_qw3.with_structured_output(GradeDocuments)

# 提示词模板
system = """
    你是一个评估检索文档与用户问题相关性的评分器。\n 
    如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n
    不需要非常严格的测试，目的是过滤掉错误的检索结果。\n
    给出'yes'或'no'的二元评分来表示文档是否与问题相关。
    请严格按照以下JSON格式返回结果： {{\"binary_score\":\"yes或no\"}}
"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("user", "Retrieved document: \n\n {document} \n\n User question: {question}") # 用户输入模板
])

# 构建检索器评分工作流
retrieval_grader_chain =  grade_prompt | structured_llm_grader # 组合提示模板和LLM评分器
