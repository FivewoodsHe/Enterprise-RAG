# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings, ZhipuAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.env_utils import DEEPSEEK_API_KEY, ZHIPU_API_KEY

# 通义千问
qw_embeddings = DashScopeEmbeddings(
    dashscope_api_key=DEEPSEEK_API_KEY,
    model="text-embedding-v4",
)

# 智谱AI
zp_embedding = ZhipuAIEmbeddings(
    api_key=ZHIPU_API_KEY,
    model='Embedding-3'
)

model_name = r"E:\hf_models\models\bge-small-zh-v1.5"
# cuda：调用GPU计算；cpu：调用CPU计算
model_kwargs = {"device": "cuda"}
# 是否归一化。生成的嵌入向量会被归一化为单位向量。这意味着每个嵌入向量的 L2 范数（欧几里得长度）将被缩放到 1。
encode_kwargs = {"normalize_embeddings": True}
bge_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# vectors = bge_embedding.embed_query("这是一个中文句子")
#
# print(len(vectors))  # 512维
# print( vectors)
