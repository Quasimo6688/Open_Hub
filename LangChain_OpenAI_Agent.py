import os
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import Docx2txtLoader
from transformers import GPT2Tokenizer
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI


# 使用预训练的gpt2分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 逐字打印
def print_char_by_char(answer):
    for char in answer:
        print(char, end='', flush=True)
        time.sleep(0.01)  # 这里的0.1秒可以调整，表示每个字符打印的间隔

# 设置OpenAI 相关参数
openai_api_key = input("请输入OpenAI API Key：")
os.environ["OPENAI_API_KEY"] = openai_api_key
model_name = input("请输入OpenAI模型名称(gpt-3.5-turbo-16k-0613;gpt-4-0613)：")



# FAISS存储文件夹路径
FAISS_directory = input("请输入本地FAISS文件夹路径：")
embeddings = OpenAIEmbeddings(chunk_size= 1000)

# FAISS向量数据库
try:
    # 尝试通过FAISS文件夹路径加载向量数据库
    db = FAISS.load_local(FAISS_directory, embeddings)
except Exception as e:
    # 本地向量数据库还没有内容引导进行内容添加
    print(f"本地向量数据库为空，请先添加内容↓↓↓")
    file_path = input("请输入知识库Word文件路径：")
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    # FAISS保存
    db.save_local(FAISS_directory)
    
# Prompt模版
template = """你作为一个AI问答助手。
必须通过以下双引号内的知识库内容进行问答:
“{combined_text}”

如果无法回答问题则回复:无法找到答案
我的问题是：{human_input}

"""
# 创建Prompt模版
prompt = PromptTemplate(
    input_variables=["combined_text","human_input"], 
    template=template
)

    
# OpenAI 模型对象    
llm = ChatOpenAI(temperature=0, model=model_name,max_tokens=8000)

# 创建LLMChain为基础的知识库问答工具 - 工具1
gpt_chain = LLMChain(
    llm=llm, 
    prompt=prompt, 
    verbose=True, 
)
def ask_faiss_db_with_gpt(question):
    docs = db.similarity_search(question , k=10)
    # 移除每个匹配项中的特殊字符（如换行符和多余的空格），并按score顺序连接标题和内容，同时保证总token不超过3000
    cleaned_matches = []
    total_tokens = 0
    for context in docs:
        clean_context = context.page_content.replace('\n', ' ').strip()
        clean_content = f"{clean_context}"
        tokens = tokenizer.encode(clean_content, add_special_tokens=False)
        if total_tokens + len(tokens) <= 8000:
            cleaned_matches.append(clean_content)
            total_tokens += len(tokens)
        else:
            break

    # 将清理过的匹配项组合成一个字符串
    combined_text = " ".join(cleaned_matches)
    
    answer = gpt_chain.predict(combined_text = combined_text , human_input = question)
    return answer

# 创建互联网搜索引擎工具 - 工具2
serp_api_key = input("请输入Serp API Key：")
search = SerpAPIWrapper(serpapi_api_key = serp_api_key)

# 创建Agent的工具箱
tools = [
    Tool(
        name = "Search_Engine",
        func = search.run,
        description = "你可以首先使用互联网搜索引擎工具进行信息查询，尝试直接找到问题答案. 注意你需要提出非常有针对性准确的问题。"
    ),
    Tool(
        name = "Local_FAISS_Vectorstore_Knowledge_Base",
        func = ask_faiss_db_with_gpt,
        description = "当你无法通过互联网搜索引擎找到准确答案时，比如未公开内容、无法直接访问的页面，你可以通过本地向量数据知识库尝试寻找问答答案。 注意你需要提出非常有针对性准确的问题。"
    )
]

# 创建Agent，类型OPENAI_FUNCTIONS
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

while True:
    # 用户提问
    print("\n=================会话开始(输入“退出”结束会话)==================\n")
    question =  input("我的问题：")
    if question.lower() == '退出':
        break
        
    answer = agent.run(question)
    print("答：")
    print_char_by_char(answer)


