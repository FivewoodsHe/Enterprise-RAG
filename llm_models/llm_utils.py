from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI

from utils.env_utils import QWEN_API_KEY, QWEN_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPU_API_KEY

zhipuai_client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 阿里云百炼：https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.1ee87b085tWQ9E&tab=model#/model-market
llm_qw3 = ChatOpenAI(
    model="qwen3-max",  # 指定使用的模型版本
    temperature=0.9,  # 控制生成文本的随机性，值越高越随机，0.9表示较活跃的生成风格
    # max_tokens=256,  # 设置模型生成的最大token数，此处限制为256
    # top_p=1,  # 核采样参数，1表示关闭该机制，不进行截断采样
    # frequency_penalty=0,  # 防止重复的惩罚系数，0表示不进行惩罚
    # presence_penalty=0.6,  # 鼓励模型谈论新话题的惩罚系数，0.6表示适度鼓励
    # verbose=True,  # 输出调试信息，便于观察模型调用过程
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    # enable_thinking 参数开启思考过程，QwQ 与 DeepSeek-R1 模型总会进行思考，不支持该参数
    extra_body={"enable_thinking": False},
)

llm_qw_ds = ChatOpenAI(
    model="deepseek-r1-0528",
    temperature=0.9,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL
)


llm_ds = ChatOpenAI(
    # DeepSeek-R1原版模型  模型指向 DeepSeek-R1-0528， deepseek-chat  deepseek-reasoner
    model="deepseek-chat",
    temperature=0.8,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# 全模态大模型qwen-omni-turbo
"""通义千问多模态理解生成大模型"""
llm_qw_dmt = ChatOpenAI(
    model="qwen-omni-turbo",
    temperature=0.9,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    streaming=True # 启用流式输出
)


# 网络搜索工具
web_search_tool = TavilySearchResults(max_results=2)
