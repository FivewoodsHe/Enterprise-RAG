import copy
import multiprocessing
import os
from multiprocessing.queues import Queue

from documents.Markdown_Parser import MarkdownParser
from documents.dense_insert_milvus_optimize import MilvusVectorSave
from utils.log_utils import log

"""
    多进程将md文档批量插入数据库
"""
def file_parser_process(dir_path:str, output_que:Queue, batch_size:int = 30):
    """进程1：解析所有的md文档，并分批放入队列中"""
    log.info(f'开始解析目录：{dir_path}')
    # 获取目录下所有的md文档
    md_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith('.md')]
    if not md_files:
        log.info(f'目录下没有md文件')
        output_que.put(None) # 提供None作为队列的终止信号
        return

    parser = MarkdownParser()
    doc_batch = []
    for md_file in md_files:
        try:
            docs = parser.parse_markdown_to_documents(md_file)
            if docs:
                doc_batch.extend(docs)
            if len(doc_batch) >= batch_size:
                # doc_batch类型是List,存在嵌套可变结构，在多线程中考虑使用深拷贝更稳妥。
                # 假设数据结构简单：如果 doc_batch 存储的是简单的文档对象或字符串，浅拷贝不会有问题
                output_que.put(copy.deepcopy(doc_batch))
                doc_batch.clear()

        except Exception as e:
            log.error(f"解析失败：{md_file}: {str(e)}")
            log.exception(e)

    # 若有剩余的文档，将其放入队列中
    if doc_batch:
        output_que.put(doc_batch)
    # 队列的终止信号
    output_que.put(None)
    log.info(f'解析完成，共处理{len(md_files)}完成')

def milvus_writer_process(input_queue:Queue):
    """进程2： 从队列中读取并写入Milvus"""
    log.info("Milvus写入进程启动。。。。")

    mv = MilvusVectorSave()
    mv.create_collection() # 创建集合
    mv.create_connection() # 创建连接
    total_count = 0
    while True:
        try:
            # 获取队列中的数据，如果为None则表示队列已结束
            datas = input_queue.get()
            if datas is None:
                break

            if isinstance(datas, list):
                mv.add_docs(datas)
                total_count += len(datas)
                print(f"写入Milvus成功，当前已写入{total_count}条数据")

        except Exception as e:
            log.error(f"写入Milvus失败：{str(e)}")
            log.exception(e)

    log.info(f"写入进程完成，总计写入{total_count}条数据")



if __name__ == '__main__':
    # 配置参数
    md_dir = r'D:\HNPython\00msb资料\04RAG企业级\md'
    queue_max = 30 # 设置队列最大长度，防止内存溢出

    # mv = MilvusVectorSave()
    # mv.create_collection(is_first=True)

    # 创建进程间的通信队列
    process_queue = Queue(maxsize=queue_max, ctx=multiprocessing.get_context())

    # 启动子进程，解析md；写入milvus。
    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(md_dir, process_queue)
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(process_queue,)
    )

    parser_proc.start()
    writer_proc.start()

    # 等待进程结束
    parser_proc.join()
    writer_proc.join()
    print("===========success所有进程结束=========")
