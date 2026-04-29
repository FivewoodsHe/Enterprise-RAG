from pprint import pprint

from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, DataType, Function, FunctionType, RRFRanker, AnnSearchRequest

from documents.Markdown_Parser import MarkdownParser
from documents.dense_insert_milvus_optimize import MilvusVectorSave
from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MILVUS_URI, COLLECTION_NAME


def test2():
    """
        这是一个测试demo：单独的全文检索测试，使用pymilvus
        创建一个新的collection: full_demo
        pymilvus自带的代码可以设置schema配置，但是langchain-milvus没有开放schema设置。
    """
    client = MilvusClient(uri=MILVUS_URI)

    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field('text',datatype=DataType.VARCHAR, max_length=10000, enable_analyzer=True,
                     # analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
                     analyzer_params={"type":"chinese"})  # 中文分词器的两种写法。
    # schema.add_field('text',datatype=DataType.VARCHAR, max_length=10000, enable_analyzer=True)  # 开启分词器，默认分词器为英文分词器
    schema.add_field('category', datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field('sparse', datatype=DataType.SPARSE_FLOAT_VECTOR, )

    bm25_function = Function(
        name = 'text_sparse',
        input_field_names=['text'],
        output_field_names='sparse',
        function_type=FunctionType.BM25
    )
    schema.add_function(bm25_function)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name='sparse',
        index_name='sparse_index',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type='BM25',
        params={
            'inverted_index_algo': 'DAAT_MAXSCORE',
            'bm25_k1': 1.2,
            'bm25_b1': 0.75
        }
    )

    collection_name = 'full_demo'
    if collection_name in client.list_collections():
        client.release_collection(collection_name)
        client.drop_index(collection_name, 'sparse_index')
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

def test3():
    """
        向具有中文分词器集合中插入数据
    """
    vector_store = Milvus(
        embedding_function=None,
        collection_name='full_demo',
        builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
        vector_field='sparse',
        consistency_level='Strong',
        auto_id=True,
        connection_args={'uri': MILVUS_URI}
    )

    file = r'D:\HNPython\MyRAGDemo\datas\md\tech_report_0yh54uvm.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file)

    vector_store.add_documents(docs)


def query_test4():
    """使用langchain-milvus进行全文搜索"""
    vector_store = Milvus(
        embedding_function=None,
        collection_name='full_demo',
        builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
        vector_field='sparse',
        connection_args={'uri': MILVUS_URI}
    )

    res = vector_store.similarity_search(
        query='善约',   # 这里如果没用中文分词器，真的会搜索不到结果。确认。如果分词器没有分到词，就搜索不到结果。'善约'
        k=3,
        expr='category=="content"'
    )

    for res in res:
        print(res.page_content + "\n")

def query_test5():
    """使用pymilvus进行搜索"""
    client = MilvusClient(uri=MILVUS_URI)

    res = client.search(
        collection_name='full_demo',
        data=['栅极电场'],
        output_fields=['text', 'category'],
        anns_field='sparse', #  (Approximate Nearest Neighbor Search Field) 指定用于近似最近邻搜索的向量字段。显式指定以提高代码可读性和避免歧义。当集合包含多个向量字段时，必须明确指定 anns_field，否则会抛出异常。
        limit=3,
        search_params={
            'param' : {'drop_ratio_search': 0.2}  # todo 这里的参数，有可能是通过阅读源码得到的。
        }
    )

    for item in res[0]:
        pprint(item)

def query_test6():
    """混合检索：pymilvus进行搜索"""
    client = MilvusClient(uri=MILVUS_URI)

    """
         FAQ: nprobe参数控制在搜索过程中要访问的倒排列表（inverted lists）或聚类（clusters）的数量。
         该数值与搜索的准确度成正比，与搜索速度成反比。在密集向量中广泛使用。
    """

    param1 = {
        'data': [bge_embedding.embed_query('最先进的纳米级清洗技术是什么？')],
        'anns_field': 'dense',
        'param': {
            'metric_type': 'IP',
            'params': {'nprobe': 10}
        },
        'limit': 5
    }
    req1 = AnnSearchRequest(**param1)

    param2 = {
        'data': ['纳米级清洗技术是什么？'],
        'anns_field': 'sparse',
        'param': {
            'metric_type': 'BM25'
        },
        'limit': 5
    }
    # FAQ: ** 操作符用于解包字典作为函数参数.不使用 **会将整个字典作为位置参数传递给函数。
    req2 = AnnSearchRequest(**param2)

    res= client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[req1, req2],
        ranker=RRFRanker(60),
        limit=5,
        output_fields=['text', 'category', 'title'],
    )

    for item in res:
        print('top5的结果是：\n')
        for i in item:
            pprint(i)


def query_test7():
    """混合检索：langchain-milvus进行搜索"""
    mv = MilvusVectorSave()
    mv.create_connection()

    res = mv.vector_store_saved.similarity_search(
        query='最先进的纳米清洗技术是什么？',
        k = 3,
        ranker_type='rrf',
        ranker_params={"k": 80}
    )

    for item in res:
        print(item.page_content + "\n")


def query_test8():
    """混合检索：langchain-milvus进行搜索"""
    mv = MilvusVectorSave()
    mv.create_connection()

    # 转换为检索器,链式调用
    retriever = mv.vector_store_saved.as_retriever(
        # similarity：返回相似度最高的结果。  mmr：返回多样性最高的结果，由于有归一化处理这里不能用mmr。
        search_type='similarity',
        search_kwargs={
            'k': 3,
            'score_threshold': 0.1,
            'ranker_type': 'rrf',
            'ranker_params': {"k": 100},
            'filter': {'category': 'content'}
        }
    )

    res = retriever.invoke('最先进的纳米清洗技术是什么？')
    for doc in res:
        print(doc)
        print('\n')


if __name__ == '__main__':
    # test2() # 1.创建collection
    #
    # test3() # 2. 插入数据

    # query_test4() # 3. 查询数据

    # query_test5() # pymilvus进行搜索

    # query_test6()

    # query_test7() # langchain-milvus进行混个检索

    query_test8() # langchain-milvus的retriever对象进行混个检索，链式调用



