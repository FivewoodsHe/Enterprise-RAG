from langchain_community.embeddings import ZhipuAIEmbeddings, DashScopeEmbeddings
from langchain_openai import OpenAIEmbeddings


# openai_embedding = OpenAIEmbeddings(
#     openai_api_key="b516ed05d67642b7964d47856e6b1b6b.zdU1c89EGHmJxFhG",
#     openai_api_base="https://open.bigmodel.cn/api/paas/v4/chat/completions",
#     model= "embedding-3"
# )


# embeddings = ZhipuAIEmbeddings(
#     api_key="b516ed05d67642b7964d47856e6b1b6b.zdU1c89EGHmJxFhG",
#     api_base="https://open.bigmodel.cn/api/paas/v4/chat/completions",
#     model= "embedding-3"
# )
#
# vecs = embeddings.embed_query('这是个文本呢')
# print(vecs[:10])


# 嵌入模型测试
emb = DashScopeEmbeddings(
    dashscope_api_key="sk-6ce48924f293474187d29af19977f4f0",
    model="text-embedding-v4",
)

vecs = emb.embed_query('这是个文本呢')
print(vecs[:10])