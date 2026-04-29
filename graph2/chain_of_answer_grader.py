from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_models.llm_utils import llm_qw3


class GradeAnswer(BaseModel):
    """评估回答是否解决用户问题的二元评分模型"""
    binary_score: str = Field(description="回答是否解决了问题，取值为'yes'或'no'")


# 提示词模板
system = """您是一个评估回答是否解决用户问题的评分器。\n
     给出'yes'或'no'的二元评分。'yes'表示:回答确实解决了该问题。
     请严格按照以下JSON格式返回结果： {{\"binary_score\":\"yes或no\"}}
     """
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统角色设定
        ("human", "用户问题: \n\n {question} \n\n 生成回答: {generation}"),  # 用户输入模板
    ]
)

# 构建回答质量评估工作流
answer_grader_chain = (
        answer_prompt  # 使用回答评估提示模板
        | llm_qw3.with_structured_output(GradeAnswer)  # 调用结构化评分的LLM
)


