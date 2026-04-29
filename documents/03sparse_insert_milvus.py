from pprint import pprint

from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(uri='http://192.168.91.65:19530')



# 删除已创建数据库
# client.drop_collection(collection_name='demo_emb')
# print('已删除数据库')

# 释放重新加载
# 将集合从内存中卸载，释放占用的资源，通常在不需要频繁查询或需要节省内存时使用
# client.release_collection("demo_emb")
# 只有加载到内存的集合才能进行搜索和查询操作。可以提高查询性能，但会占用更多内存资源
# client.load_collection("demo_emb")

"""
Schema：是集合的"蓝图"或"设计图"，描述集合应该包含哪些字段及属性.
Milvus 根据 schema 中的字段定义创建对应的物理存储结构,每个字段按照 schema 中的配置进行初始化（如数据类型、是否为主键等）
"""
schema = client.create_schema()
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
# milvus自带en分词器，中文分词器需要手动添加
schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=2000, enable_analyzer=True)
schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR) # 存储稀疏向量


# 稀疏嵌入的函数
bm25_function = Function(
    name="text_bm25_emb2", description="BM25向量函数",
    input_field_names=['text'],
    output_field_names=['sparse'],
    function_type=FunctionType.BM25
)

schema.add_function(bm25_function)

# 配置索引
"""
inverted_index_algo: 指定倒排索引的算法类型。
    - DAAT_MAXSCORE：文档至少包含一个查询词，计算最大分值（推荐）
    - DAAT_WAND：使用 WAND 算法优化性能。
    - TAAT_NAIVE：基础的词优先搜索算法。
"bm25_k1":  BM25 算法的饱和度参数，控制词频对相关性得分的影响程度，值越大，高频词的影响越显著（通常设置在 1.2-2.0 之间）。
"bm25_b":BM25 算法的长度归一化参数，控制文档长度对相关性得分的影响，值为 0 表示不考虑长度因素，值为 1 表示完全考虑（标准值为 0.75）。
"""
index_params = client.prepare_index_params()
index_params.add_index(
    field_name='sparse',
    index_name='sparse_inverted_index',
    index_type='SPARSE_INVERTED_INDEX',
    metric_type='BM25',
    params={
        "inverted_index_algo" : "DAAT_MAXSCORE", # 用于构建和查询索引的算法。有效值：DAAT_MAXSCORE、DAAT_WAND、TAAT_NAIVE。
        "bm25_k1" : 1.6,
        "bm25_b" : 0.75
    }
)

def create_coll():
    # 创建一张表
    client.create_collection(
        collection_name='demo_emb',
        schema=schema,
        index_params=index_params
    )

def insert_data():
    # 插入测试数据：
    client.insert(
        collection_name='demo_emb',
        data = [
            {"text": "I like milk tea"},
            {"text": "I like coffea"},
            {"text": "I like basketball"},
            {'text': 'information retrieval is a field of study.'},
            {'text': 'information retrieval focuses on finding relevant information in large datasets.'}, # 信息检索的重点是在大型数据集中查找相关信息。
            {'text': 'data mining and information retrieval overlap in research.'}, # 数据挖掘和信息检索在研究中是重叠的。
        ]
    )

def search_data():
    search_params = {
        "params" : {'drop_ratio_search' : 0.2 } # 搜索时要忽略的低重要性词语的比例：查询中向量中最小的20%值的词语将被忽略
    }

    resp = client.search(
        collection_name='demo_emb',
        data= ['whats the focus of information retrieval??'],
        anns_field='sparse',
        limit=2,
        search_params=search_params,
        output_fields=['text']
    )
    pprint( resp)

if __name__ == '__main__':
    # create_coll()
    # insert_data()
    search_data()


