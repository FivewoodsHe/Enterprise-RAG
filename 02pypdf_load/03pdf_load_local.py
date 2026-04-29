import json
from pprint import pprint

from langchain_unstructured import UnstructuredLoader

pdf_file = r'/home/pythonTest/layout-parser-paper.pdf'
"""
使用虚拟机上私有化unstructured的api进行pdf解析
"""

def write_json(data, file_name):
    with open('/home/pythonTest/output/' + file_name, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


loader = UnstructuredLoader(
    file_path=pdf_file,
    strategy='hi_res',
    partition_via_api=False
)

docs = []
counter = 0
for doc in loader.lazy_load():
    docs.append(doc)
    json_file_name = str(doc.metadata.get('page_number')) + '_' + str(counter) + '.json'
    counter += 1
    write_json(doc.model_dump(), json_file_name)

print(f'doc的数量是: {len(docs)}')
print('第一个doc是：')
pprint(docs[0].metadata)
pprint(docs[0].page_content)

print('--' * 50)


segments = [
    doc.metadata
    for doc in docs
    if doc.metadata.get("page_number") == 5 and doc.metadata.get("category") == "Table"
]
print(f'表格数据为:')
print(segments)
