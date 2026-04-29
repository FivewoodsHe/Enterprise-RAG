from langchain_core.tools import create_retriever_tool

from documents.dense_insert_milvus_optimize import MilvusVectorSave

mv = MilvusVectorSave()
mv.create_connection()
retriever = mv.vector_store_saved.as_retriever(
    search_type='similarity',  # 设置相似度搜索，仅返回相似度超过阈值的文档。
    search_kwargs = {
        "k": 4, # 最多返回的文档数量
        "score_threshold": 0.1, # 设置相似度得分最低的阈值，低于该阈值的文档将被忽略
        "ranker_type": "rrf", # 使用倒数排名融合方法对结果进行重排序。
        "ranker_params": {"k": 100}, # 为RRF提供额外的参数，这里的K是公式中的常数。
        # 'filter': {"category": "content"}  # 添加过滤条件，优先过滤掉不符合条件的文档。
    }
)

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='rag_retriever',
    description='搜索并返回关于 ‘半导体和芯片’ 的信息, 内容涵盖：半导体和芯片的封装、测试、光刻胶等'
)