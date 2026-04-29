import os

from dotenv import load_dotenv

"""
读取项目根目录下的 .env 文件
将文件中定义的键值对设置为环境变量
使得应用程序可以通过 os.getenv() 等方式获取这些变量

load_dotenv()：加载 .env 文件中的环境变量
override=True：如果环境变量已经存在，则用 .env 文件中的值覆盖

优势:
    环境隔离：不同环境（开发、测试、生产）可以使用不同的配置
    安全保护：敏感信息（如 API 密钥）不会硬编码在代码中
    便于管理：集中管理配置信息，易于修改和维护
    团队协作：每个开发者可以有自己的 .env 配置文件
"""

load_dotenv(override=True)

# deepseek
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL=os.getenv("DEEPSEEK_BASE_URL")

# 通义千问
QWEN_API_KEY=os.getenv("QWEN_API_KEY")
QWEN_BASE_URL=os.getenv("QWEN_BASE_URL")

# 智谱ai
ZHIPU_API_KEY=os.getenv("ZHIPU_API_KEY")

# 高德地图
GAODE_API_KEY=os.getenv("GAODE_API_KEY")

# 塔维
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

# Milvus数据库地址
MILVUS_URI = 'http://192.168.91.65:19530'
# 集合名称
COLLECTION_NAME = 't_md'
