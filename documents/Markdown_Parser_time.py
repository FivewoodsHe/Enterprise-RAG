import time
from pprint import pprint
from typing import List


from langchain_experimental.text_splitter import SemanticChunker

from llm_models.embeddings_model import  bge_embedding
from utils.log_utils import log

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

from utils.time_utils import time_counter


class MarkdownParser:
    """专门用于markdown文档解析和切片"""
    # SemanticChunker会调用hf的bge模型，hf初始化会从hf官网上下载数据库，所以会很慢。
    # - 因此要么配置hf的国内镜像网站、要么开启翻墙代理。或者是设置成使用时加载
    def __init__(self):
        self.text_splitter_m = None

    def _get_text_splitter(self):
        """获取或初始化text_splitter_m属性"""
        if self.text_splitter_m is None:
            print("正在初始化text_splitter_m属性...")
            self.text_splitter_m = SemanticChunker(
            embeddings = bge_embedding,
            breakpoint_threshold_type='percentile'
        )
        return self.text_splitter_m

    @time_counter
    def text_chunker(self, datas: List[Document]) -> List[Document]:
        """
        对markdown文档进行切片
        样式： 一级标题 -> 文本内容 二级标题 -> （文本内容） 三级标题 -> 文本内容
        """
        new_docs = []

        # 避免加载hf,判断是否有长文档
        long_flag = False
        for doc in datas:
            if  len(doc.page_content) > 5000:
                long_flag = True
                break # 跳出最内层的while/for循环
        text_splitter = None
        if long_flag:
            text_splitter = self._get_text_splitter()

        for doc in datas:
            if  len(doc.page_content) > 5000:
                # 通过从可迭代对象中添加元素来扩展列表。
                new_docs.extend(text_splitter.split_documents([doc]))
                continue
            # 将对象追加到列表的末尾
            new_docs.append(doc)
        return new_docs

    @time_counter
    def parse_markdown_to_documents(self, md_file: str, encoding='utf-8') -> List[Document]:
        documents = self.load_markdown(md_file)
        log.info(f'解析后的长度为：{len(documents)}')

        # 进行文本合并
        merged_documents = self.merge_title_content(documents)
        log.info(f'合并后的长度为：{len(merged_documents)}')

        # log.info('*' * 50)
        #
        # for i in range(len(merged_documents)):
        #     pprint(merged_documents[i].metadata)
        #     print(f"\n标题: {merged_documents[i].metadata.get('title', None)}")
        #     print(f"doc的内容: {merged_documents[i].page_content}\n")
        #     print("------" * 10)
        #
        # log.info('*' * 50)

        # 长度超过1000的进行语义切割
        chunk_documents = self.text_chunker(merged_documents)
        log.info(f'语义切割后的长度为：{len(chunk_documents)}')
        return chunk_documents

    @staticmethod
    @time_counter
    def load_markdown(md_file: str) -> List[Document]:
        """加载读取markdown文档"""
        loader = UnstructuredMarkdownLoader(
            file_path=md_file,
            mode='elements',
            strategy='fast'
        )
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)
        return docs

    @staticmethod
    @time_counter
    def merge_title_content(datas: List[Document]) -> List[Document]:
        """
        将title和content进行合并
        """
        merged_data = []
        parent_dict = {} # 保存所有父document,key为document的id,value为document
        for document in datas:
            metadata = document.metadata
            # 删除languages字段，是因为metadata中，值为list，在后续dense_insert_milvus.py中处理会报错
            if 'languages' in metadata:
                metadata.pop('languages')

            element_id = metadata.get('element_id', None)
            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)

            # 特殊处理，文章标题之前或其他位置的文字，属于叙述文本，但是不在标题之下
            if category == 'NarrativeText' and parent_id is None:
                merged_data.append(document)

            if category == 'Title': # 标题类的document放入父类字典中
                document.metadata['Title'] = document.page_content
                # 如果有父类，将父类内容添加到子类中
                if parent_id in parent_dict:
                    document.page_content = parent_dict[parent_id].page_content + ' -> ' + document.page_content
                parent_dict[element_id] = document

            if category != 'Title' and parent_id:
                parent_dict[parent_id].page_content += ' ' + document.page_content
                parent_dict[parent_id].metadata['category'] = 'content'

        if parent_dict is not None:
            merged_data.extend(parent_dict.values())
        return merged_data


if __name__ == '__main__':
    s1_time = time.time()
    file_path = r'D:\HNPython\MyRAGDemo\datas\md\tech_report_0yh54uvm.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)
    s2_time = time.time()
    e1_time = s2_time - s1_time
    print(f'md文档已解析，长度为： {len(docs)}，解析耗时为：{e1_time:.2f}秒\n')

    for i in range(int(len(docs)/2)):
        pprint(docs[i].metadata)
        print(f"\n标题: {docs[i].metadata.get('title', None)}")
        print(f"doc的内容: {docs[i].page_content}\n")
        print("------" * 10)














