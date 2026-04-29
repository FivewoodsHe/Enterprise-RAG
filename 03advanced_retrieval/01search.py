from documents.dense_insert_milvus import MilvusVectorSave


def test1():
    mv = MilvusVectorSave()
    mv.create_connection(False)

    # ANN近似紧邻算法，在KNN的基础上优化，
    result = mv.vector_store_saved.similarity_search_with_score(
        query="最先进的芯片封装技术有哪些？",
        k = 3,
        expr="category=='content'"
    )
    # similarity_search_with_score返回的是一个列表，列表中每个元素是一个元组，元组中第一个元素是文档，第二个元素是相似度
    for res in result:
        print(res)
        print("\n")




if __name__ == '__main__':
    test1()