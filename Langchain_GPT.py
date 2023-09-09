import faiss
import numpy as np
import openai
from transformers import GPT2Tokenizer
import time
import os

# 使用预训练的gpt2分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 定义一个全局变量作为调试标志
DEBUG = False

# 请填写您的OpenAI API密钥，如果已保存在本地文件中，请读取
api_key_file_path = "D:\\AI\\pypro\\key.TXT"
try:
    with open(api_key_file_path, "r") as key_file:
        api_key = key_file.read().strip()
except FileNotFoundError:
    api_key = input("请输入您的OpenAI API密钥：")

openai.api_key = api_key

# 设置请求延迟时间，根据GPT的访问规则进行设置
REQUEST_DELAY_SECONDS = 2  # 请根据需要调整延迟时间

# 逐字打印
def print_char_by_char(answer):
    for char in answer:
        print(char, end='', flush=True)
        time.sleep(0.1)  # 这里的0.1秒可以调整，表示每个字符打印的间隔

# 将问题转换为密集向量表示
def create_embedding(text):
    model_engine = "text-embedding-ada-002"
    response = openai.Embedding.create(
        model=model_engine,
        input=text,
    )
    return response['data'][0]['embedding']

# 加载本地Faiss索引
faiss_index_path = "D:\\AI\\pypro\\faiss_index.bin"
index = faiss.read_index(faiss_index_path)

while True:
    # 用户提问
    print("=================会话开始(输入“退出”结束会话)==================\n")
    question = input("问：")

    if question.lower() == '退出':
        break

    # 将问题文本转换为向量
    question_embedding = create_embedding(question)

    # 列举查询匹配最靠前10的文本
    D, I = index.search(np.array([question_embedding]), 10)

    # 获取metadata中的Title和LangChain_context
    # 这里需要您提供具体的代码来获取metadata，因为它取决于您的Faiss索引的构建方式
    # 请提供获取metadata的代码，以便我帮助您集成到代码中

    if DEBUG:
        # 获取debug_metadata中的信息
        # 请提供获取debug_metadata的代码，以便我帮助您集成到代码中

    # 移除每个匹配项中的特殊字符（如换行符和多余的空格），并按score顺序连接标题和内容，同时保证总token不超过3000
    cleaned_matches = []
    total_tokens = 0
    for title, context in top_matches:
        clean_context = context.replace('\n', ' ').strip()
        clean_content = f"{title} {clean_context}"
        tokens = tokenizer.encode(clean_content, add_special_tokens=False)
        if total_tokens + len(tokens) <= 3000:
            cleaned_matches.append(clean_content)
            total_tokens += len(tokens)
        else:
            break

    # 将清理过的匹配项组合成一个字符串
    combined_text = " ".join(cleaned_matches)

    # 拼接 Prompt
    prompt = f"必须通过以下双引号以内的开发文档内容进行问答:\n“{combined_text}”\n\n如果无法回答问题则回复:无法找到答案\n我的问题是：{question}"

    if DEBUG:
        # 请提供获取debug_metadata的代码，以便我帮助您集成到代码中

    # 控制请求频率
    time.sleep(REQUEST_DELAY_SECONDS)

    def ask_gpt(prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你作为一个LangChain技术开发助手\n"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            n=1,
            temperature=0,
        )
        return response.choices[0].message['content'].strip()

    answer = ask_gpt(prompt)
    print("答：")
    print_char_by_char(answer)
    print("\n=================会话结束(输入“退出”结束会话)==================\n")
