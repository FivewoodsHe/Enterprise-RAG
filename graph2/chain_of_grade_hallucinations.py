from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_models.llm_utils import llm_qw3


# 数据模型 - 生成内容幻觉评分
class GradeHallucinations(BaseModel):
    """对生成回答中是否存在幻觉进行二元评分"""
    binary_score: Literal["yes", "no"] = Field(description="回答是否基于事实，取值为'yes'或'no'")



# 提示词模板
system = """"
    您是一个评估生成内容是否基于检索事实的评分器。\n
    给出'yes'或'no'的二元评分。'yes'表示回答是基于/支持于给定事实集的。
    请严格按照以下JSON格式返回结果： {{\"binary_score\":\"yes或no\"}}
"""
# hallucination 翻译为 "幻觉" 或 "虚假信息"。它指的是评估模型回答是否基于真实检索到的事实，还是产生了"幻觉"
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system), # 系统角色设定
    ("human", "请对以下内容进行评估：\n\n 事实集：{text} \n\n 生成内容：{generation}"), # 用户输入模板
])

# 构建幻觉检测工作流 # 使用幻觉检测提示模板
hallucination_grader_chain = hallucination_prompt | llm_qw3.with_structured_output(GradeHallucinations)