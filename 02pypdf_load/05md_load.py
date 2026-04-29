from pprint import pprint

from langchain_community.document_loaders import UnstructuredMarkdownLoader


"""
通过unstructured[md]的对md文档进行解析
总结：
1.会按照段落来；
2.标题和正文之间有父子关系。
"""


mdLoder = UnstructuredMarkdownLoader(
    file_path=r"D:\HNPython\MyRAGDemo\datas\md\product_faq.md",
    mode="elements"
)

docList = mdLoder.load()

print(f"解析后文档长度：{len(docList)}")

for i in range(20):
    pprint(docList[i].metadata)
    print(docList[i].page_content)
    print("-" * 50)
