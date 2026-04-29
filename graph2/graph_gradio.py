from pprint import pprint
from typing import List, Dict

import gradio as gr


from graph2.my_graph2 import graph

# faq: 第二个参数命名为chatbot时，会提示："从外部作用域隐藏名称 'chatbot'"。这是因为：变量命名冲突导致的作用域遮蔽问题，函数参数 chatbot 遮蔽了外层的组件变量 chatbot
def do_graph(input_text, chat_history):
    """输入框提交后执行的函数"""
    if input_text:
        chat_history.append({"role": "user", "content": input_text})
    return  '', chat_history



def execute_graph(chat_bot:List[Dict]) -> List[Dict]:
    """ 执行工作流的函数"""
    user_input = chat_bot[-1]['content']
    result = '' # AI助手的最后一条消息

    inputs = {"question": user_input}

    resp = graph.stream(inputs)
    print(resp)

    # 流式执行工作流
    for output in resp:
        for key, value in output.items():
            # 打印当前节点名称
            pprint(f"Node '{key}':")
            pprint("\n---\n")

    # 打印最终结果
    result = value["generation"]
    pprint(result)

    chat_bot.append({"role":"assistant", "content": result})
    return chat_bot

css = '''
#bgc {background-color: #7FFFD4}
.feedback textarea {font-size: 24px !important}
'''
with gr.Blocks(css=css, title="混合检索+自评估RAG") as instance:
    gr.Label("混合检索+自评估RAG", container=False)
    chatbot = gr.Chatbot(type="messages", height=350, label='AI客服') # 聊天记录组件

    input_textbox = gr.Textbox(label="请输入问题", placeholder="here")

    input_textbox.submit(do_graph, [input_textbox, chatbot], [input_textbox, chatbot]).then(execute_graph, chatbot, chatbot)


if __name__ == '__main__':
    instance.launch()



