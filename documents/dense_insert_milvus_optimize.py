import time
from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, DataType, Function, FunctionType, IndexType
from pymilvus.client.types import MetricType
from unstructured_client.utils.serializers import is_nullable

from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MILVUS_URI, COLLECTION_NAME
from documents.Markdown_Parser import MarkdownParser

"""
    优化版本：使用pymilvus创建集合，创建schema时可以设置指定字段使用中文分词器。
"""


class MilvusVectorSave:
    """ 把文件md，sparse和dense embedding 插入到milvus数据库中 """

    def __init__(self) -> object:
        """延迟初始化: 在实际需要时才创建和保存 Milvus连接实例；
        状态管理: 通过该变量跟踪是否已经建立了向量存储连接；
        复用连接: 避免重复创建连接，提高性能
        """
        self.vector_store_saved: Milvus = None

    def create_collection(self):
        """使用pymilvus模块创建collection集合，给text字段设置中文分词器"""
        client = MilvusClient(uri=MILVUS_URI)
        schema = client.create_schema()
        schema.add_field('id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field('text', datatype=DataType.VARCHAR, max_length=10000, enable_analyzer=True, analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
        schema.add_field('category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000, is_nullable=True)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512)

        bm25_function = Function(
            name='text_sparse',
            input_field_names=['text'],
            output_field_names='sparse',
            function_type=FunctionType.BM25
        )
        schema.add_function(bm25_function)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name='sparse',
            index_name='sparse_vector_ind',
            index_type='SPARSE_INVERTED_INDEX',
            # 使用BM25进行相似度计算
            metric_type='BM25',
            params={
                'inverted_index_algo': 'DAAT_MAXSCORE',
                'bm25_k1': 1.2,  # 范围[1.2 ~ 2.0]
                'bm25_b': 0.75
            },
        )
        index_params.add_index(
            # 稠密向量索引
            field_name='dense',
            index_name='dense_vector_ind',
            index_type=IndexType.HNSW,
            metric_type=MetricType.IP,
            # 近似相邻算法的调优参数
            params={"M": 16, "efConstruction": 64}
        )

        if COLLECTION_NAME in client.list_collections():
            # 删除索引/集合，先让内存释放。需要判重，否则会报错
            client.release_collection(COLLECTION_NAME)
            client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_vector_ind')
            client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_vector_ind')
            client.drop_collection(COLLECTION_NAME)

        client.create_collection(
            schema=schema,
            index_params=index_params,
            collection_name=COLLECTION_NAME,
        )

    def create_connection(self):
        """创建 Milvus 连接实例"""
        self.vector_store_saved = Milvus(
            embedding_function=bge_embedding,
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],  # 顺序和索引参数一致
            auto_id=True,
            connection_args={"uri": MILVUS_URI}
        )

    def add_docs(self, datas: List[Document]):
        """将文件转换稀疏、稠密向量保存在milvus中"""
        self.vector_store_saved.add_documents(datas)


if __name__ == '__main__':
    # # md文件解析
    file_path = r'D:\HNPython\MyRAGDemo\datas\md\tech_report_0yh54uvm.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    print(f'md文档已解析，长度为： {len(docs)}\n')

    # 向量保存Milvus
    print('milvus main ....')
    mv = MilvusVectorSave()
    mv.create_collection()
    mv.create_connection()

    mv.add_docs(docs)



    # # 结果验证
    # my_client = mv.vector_store_saved.client
    # # 得到表结构
    # desc_collection = my_client.describe_collection(COLLECTION_NAME)
    # print(f'表结构为：\n {desc_collection}')
    #
    # # 所有的索引
    # index_list = my_client.list_indexes(COLLECTION_NAME)
    # print(f'索引列表为：\n {index_list}')
    # if index_list:
    #     for ind_name in index_list:
    #         # 得到索引的描述
    #         index_desc = my_client.describe_index(COLLECTION_NAME, index_name=ind_name)
    #         print(f'索引描述为: {index_desc}')
    #
    # result = my_client.query(
    #     collection_name=COLLECTION_NAME,
    #     filter="category == 'Title'",
    #     output_fields=['text', 'category', 'filename']
    # )
    # print(f'测试过滤查询 结果为：\n {result}')
