from pprint import pprint

from langchain_community.document_loaders import PyPDFLoader

pdf_path= r'D:\HNPython\MyRAGDemo\datas\layout-parser-paper.pdf'
loader = PyPDFLoader(file_path=pdf_path)

documents = loader.load()

print(f'doc的数量：{len(documents)}')

pprint(documents[0].metadata)
# 'page': 0,   page下标
# 'page_label': '1',  页码

print('-----------------')
print(documents[0].page_content)