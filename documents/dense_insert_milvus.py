import time
from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus,BM25BuiltInFunction
from pymilvus import MilvusClient, DataType, Function, FunctionType, IndexType
from pymilvus.client.types import MetricType

from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MILVUS_URI, COLLECTION_NAME
from documents.Markdown_Parser import MarkdownParser


"""
    这个版本使用langchain-milvus创建集合，所以text字段没有中文分词器。
"""
class MilvusVectorSave:
    """ 把文件md，sparse和dense embedding 插入到milvus数据库中 """
    def __init__(self) -> object:
        """延迟初始化: 在实际需要时才创建和保存 Milvus连接实例；
        状态管理: 通过该变量跟踪是否已经建立了向量存储连接；
        复用连接: 避免重复创建连接，提高性能
        """
        self.vector_store_saved:Milvus = None
        print("初始化")
        """
        密集向量：
            # IndexType.HNSW 一种基于图的近似最近邻搜索算法。高精度中等大小数据库，中小型项目首选。
            # 使用内积 (Inner Product) 进行相似度计算,IP 通常用于衡量两个向量在方向上的相似性（点积越大表示越相似）。 MetricType.IP
            # M: 邻接节点数，控制节点连接的点数[4,64]，影响搜索精度。值和时间资源成正比。  # efConstruction: 搜索范围[50,200]。  
        """
        self.index_params = [
            { # 稠密向量索引
                "field_name" : 'dense',
                "index_name" : 'dense_vector_ind',
                "index_type" : IndexType.HNSW,
                "metric_type" : MetricType.IP,
                # 近似相邻算法的调优参数
                "params" : { "M": 16, "efConstruction": 64 }
            },
            { # 稀疏向量索引
                "field_name" : 'sparse',
                "index_name" : 'sparse_vector_ind',
                "index_type" : 'SPARSE_INVERTED_INDEX',
                # 使用BM25进行相似度计算
                "metric_type" : 'BM25',
                "params" : {
                    'inverted_index_algo': 'DAAT_MAXSCORE',
                    'bm25_k1': 1.2, # 范围[1.2 ~ 2.0]
                    'bm25_b1': 0.75
                },
           }
        ]

    def create_connection(self, is_create_collection=True):
        """创建milvus连接，如果是True表示需要创建集合，需要将之前的释放删除。False则表示，连接指定集合的数据库连接。"""
        if is_create_collection:
            mclient = MilvusClient(uri=MILVUS_URI)
            if COLLECTION_NAME in mclient.list_collections():
                # 删除索引/集合，先让内存释放。需要判重，否则会报错
                mclient.release_collection(COLLECTION_NAME)
                mclient.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_vector_ind')
                mclient.drop_index(collection_name=COLLECTION_NAME, index_name='dense_vector_ind')
                mclient.drop_collection(COLLECTION_NAME)

        """
        # BM25BuiltInFunction() 了启用并集成 Milvus 对 BM25 稀疏向量检索的支持
        """
        self.vector_store_saved = Milvus(
            embedding_function=bge_embedding,
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense','sparse'], # 顺序和索引参数一致
            index_params=self.index_params,
            consistency_level='Strong',  # 写入数据库立即可读，所有查询都会返回最新数据。金融等实时场景使用。
            auto_id=True,
            connection_args={"uri": MILVUS_URI}
        )

    def add_docs(self, datas:List[Document]):
        """将文件转换稀疏、稠密向量保存在milvus中"""
        self.vector_store_saved.add_documents(datas)

if __name__ == '__main__':
    s1_time = time.time()
    # # md文件解析
    file_path=r'D:\HNPython\MyRAGDemo\datas\md\tech_report_0yh54uvm.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    s2_time = time.time()
    e1_time = s2_time - s1_time
    print(f'md文档已解析，长度为： {len(docs)}，解析耗时为：{e1_time:.2f}秒\n')

    # 向量保存Milvus
    print('milvus main ....')
    mv = MilvusVectorSave()
    mv.create_connection(is_create_collection=True)
    mv.add_docs(docs)
    s3_time = time.time()
    e2_time = s3_time - s1_time
    print(f'写入数据库耗时为：{e2_time:.2f}秒 \n')

    # 结果验证
    my_client = mv.vector_store_saved.client
    # 得到表结构
    desc_collection = my_client.describe_collection(COLLECTION_NAME)
    print(f'表结构为：\n {desc_collection}')

    # 所有的索引
    index_list = my_client.list_indexes(COLLECTION_NAME)
    print(f'索引列表为：\n {index_list}')
    if index_list:
        for ind_name in index_list:
            # 得到索引的描述
            index_desc = my_client.describe_index(COLLECTION_NAME, index_name=ind_name)
            print(f'索引描述为: {index_desc}')

    result = my_client.query(
        collection_name=COLLECTION_NAME,
        filter="category == 'Title'",
        output_fields=['text','category', 'filename']
    )
    print(f'测试过滤查询 结果为：\n {result}')










