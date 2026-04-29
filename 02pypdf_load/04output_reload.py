import json
from pprint import pprint

from langchain_core.documents import Document

"""
将解析的json文件读取加载
"""

def load_doc_from_json(file_name:str):
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return Document(page_content=data['page_content'], metadata=data['metadata'])



if __name__ == '__main__':
    doc = load_doc_from_json("/datas/output/output2_16.json")
    pprint(doc)




