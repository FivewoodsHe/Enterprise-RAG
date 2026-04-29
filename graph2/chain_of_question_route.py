


"""
    查询的动态路由： 根据用户的提问，决策采用哪种检索策略（网络检索，RAG）
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_models.llm_utils import llm_qw3


# 大模型回答的数据模型
class RouteQuery(BaseModel):
    """将用户问题路由到最相关的数据源"""
    datasource: Literal['vectorstore', 'web_search'] = Field(description="根据用户问题选择将其路由到向量知识库或网络搜索")


# LLM初始化，设置结构化输出
structured_llm_router = llm_qw3.with_structured_output(RouteQuery)

# 提示词模板
system = """你是一个擅长将用户问题路由到向量知识库或网络搜索的专家。
向量知识库包含与半导体材料，芯片制造，光刻技术相关的文档,与这个有关用vectorstore标识；
对于这些主题的问题请使用向量知识库，其他情况使用网络搜索，用web_search表示。
请严格按照以下JSON格式返回结果： {{\"datasource\":\"vectorstore或web_search\"}}
"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "请将用户问题路由到最相关的数据源：{question}"),
])

# 创建问题路由器处理链
question_router_chain = route_prompt | structured_llm_router

# faq： 让大模型的输出进行制定对象转换时，提示词必须说明：请严格按照以下JSON格式返回结果： {{\"datasource\":\"vectorstore或web_search\"}}
# res1 = question_router_chain.invoke({"question": "EUV光刻机是什么？"})
# res2 = question_router_chain.invoke({"question": "如何使用向量数据库进行问答？"})
#
# print(res1)
# print(res2)



