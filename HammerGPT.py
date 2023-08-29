# 1. 导入所需的库
import os
import socket
import threading
import time
import webbrowser
import nltk
import openai
import uuid
import json
from tqdm import tqdm
from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, StringField, PasswordField, TextAreaField, FileField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import PDFMinerLoader
from transformers import GPT2Tokenizer
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from flask import jsonify
from flask import Markup

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'FileTemp') # 指定你想要保存文件的文件夹

# 使用预训练的gpt2分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# PDF分词器
nltk.download('punkt')


# 加载配置文件
try:
    with open('settings.json', 'r') as f:
        settings = json.load(f)
        temperature = settings['temp']
        model= settings['gpt_model']
        os.environ["OPENAI_API_KEY"] = settings['api_key']
        openai.api_key = settings['api_key']
except FileNotFoundError:
    print("配置文件 'settings.json' 未找到。")
except json.JSONDecodeError:
    print("'settings.json' 配置文件格式错误。")
except KeyError:
    print("'settings.json' 配置文件中缺少必要的键值。")

# 2. 创建 Flask 应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# 3. 定义下边一堆的全局变量
global_progress = 0

# OpenAI GPT + 知识库文件上传
class PromptForm(FlaskForm):
    api_key = StringField('OpenAI API KEY※:', validators=[DataRequired()])
    gpt_model = SelectField('GPT模型名称※:', choices=[('gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-16k-0613'), ('gpt-4-0613', 'gpt-4-0613')], validators=[DataRequired()])
    temp = StringField('脑洞值，0最保守(0 - 2.00):')
    doc_file = FileField('学习资料:')
    operation = StringField('Operation')
    submit = SubmitField('保存')



# 创建新的表单类来接收用户的问题
class QuestionForm(FlaskForm):
    question = TextAreaField('问题※:', validators=[DataRequired()])
    submit = SubmitField('提交')

    
# 问答Prompt模版
template = """你是一个AI智能问答机器人。
需要根据我提出的问题，结合聊天记录和知识库内容来回答我的问题
以下双引号内为聊天记录总结:
"{chat_history}"

注意：必须参考以下双引号内的知识库内容进行回答:
“{combined_text}”
如果无法回答问题再用引用你的知识辅助问答。

回答要求
1、中文回答
2、答案需要分重点进行罗列回答
3、当我的问题与聊天记录无关时，你只需要参考知识库内容进行回答
4、代码部分以MarkDown格式输出
现在我的问题是: {question}

"""


prompt = PromptTemplate(
    input_variables=["chat_history","combined_text","question"], 
    template=template
)

# 聊天记录总结Prompt模版
sum_template = """需要你根据提供的聊天记录进行总结。我会在之前的总结上增加新的聊天记录，要求你返回一个新的总结。
当前的聊天记录总结：
{summary}

新的聊天记录：
{new_lines}

注意，务必表现出整个聊天的上下文关系，以叙事的风格进行总结，一定保证问答过程的完整性。
新的聊天记录总结：
"""

sum_prompt = PromptTemplate(
    input_variables=['summary', 'new_lines'], 
    template=sum_template
)


# 使用GPT-3.5-turbo-16K，token多的用不完。但是如果有GPT-4可以在这里替换，效果更好
llm_gpt=ChatOpenAI(temperature=temperature,max_tokens=3072,model_name=model)
# 创建对话聊天总结
sum_memory = ConversationSummaryMemory(llm=llm_gpt ,memory_key="chat_history", input_key = "question")
sum_memory.human_prefix = "问题"
sum_memory.ai_prefix = "回答"
sum_memory.prompt = sum_prompt

# 创建LLMChain，
llm_chain = LLMChain(
    llm=llm_gpt, 
    prompt=prompt, 
    verbose=True, 
    memory=sum_memory,
)
# Embedding的转换，需要考虑长文章的进度
embeddings = OpenAIEmbeddings(chunk_size= 1000)
embeddings.max_retries = 100
embeddings.request_timeout = 10


# FAISS存储文件夹路径
FAISS_directory = os.path.join(os.getcwd(), 'VectorDatabase')

# 新添加的路由，用于返回学习进度
@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify(progress=global_progress)

# 定义查看资料库
@app.route('/view_library', methods=['GET'])
def view_library():
    file_list = os.listdir('FileTemp')
    return jsonify(file_list)
    
# 定义添加学习资料的路由
@app.route('/add_docs', methods=['POST'])
def add_docs():
    doc_file = request.files['doc_file']
    original_filename = doc_file.filename
    ext = os.path.splitext(original_filename)[-1]
    uuid_filename = secure_filename(f"{uuid.uuid4()}{ext}")

    filename_map = {}
    filename_map[uuid_filename] = original_filename

    # 使用os.path.join来合并路径
    file_path = os.path.join(UPLOAD_FOLDER, filename_map[uuid_filename])
    # 保存文件
    doc_file.save(file_path)

    # 获取文件扩展名
    file_extension = Path(file_path).suffix

    # 检查文件类型
    if file_extension.lower() in ['.pdf']:
        loader = PDFMinerLoader(file_path)
    elif file_extension.lower() in ['.doc', '.docx']:
        loader = Docx2txtLoader(file_path)
    else:
        print("文件类型不支持，请输入PDF或Word文件.")
        result_message = "文件类型不支持，请输入PDF或Word文件."
        return jsonify(answer = result_message)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # FAISS向量库的本地合并
    try:
        # 加载本地向量库并合并新的数据
        db_local = FAISS.load_local(FAISS_directory, embeddings)
        # 转换Embedding段落
        global global_progress
        i = 0
        for doc  in tqdm(docs):
            time.sleep(0.1)
            doc_temp = [doc]
            db_temp = FAISS.from_documents(doc_temp, embeddings)
            db_local.merge_from(db_temp)
            global_progress = (i + 1) * 100 / len(docs)  # 更新学习进度
            print("global_progress:",global_progress)
            i += 1
        # FAISS保存
        db_local.save_local(FAISS_directory)
        print("学习资料理解完毕！开始提问吧")
        result_message = "学习资料理解完毕！开始提问吧"
        time.sleep(1)
        global_progress = 0  # 重置学习进度
        return jsonify(answer = result_message)
    except Exception as e:
        # 本地向量数据库还没有内容创建添加
        # FAISS保存
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(FAISS_directory)
        print("向量库异常: ", e)
        print("本地向量数据库为空，创建并保存向量数据库")
        result_message = "本地向量数据库为空，创建并保存向量数据库"
        return jsonify(answer = result_message)


# 4. 定义应用的根路由
@app.route('/', methods=['GET', 'POST'])
def prompt_generator():
    # 从配置文件读取设置
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
            llm_gpt.model_name = settings['gpt_model']
            llm_gpt.openai_api_key = settings['api_key']
            openai.api_key = settings['api_key']
            temperature = settings['temp']
    except:
        pass
        
    prompt_form =  PromptForm(api_key=openai.api_key, gpt_model = model, temp = temperature)
    question_form = QuestionForm()
        
    # 处理参数设置并进行知识库文件Embedding转换
    if prompt_form.validate_on_submit():
        # 保存设置到配置文件
        settings = {
            'api_key': prompt_form.api_key.data,
            'gpt_model': prompt_form.gpt_model.data,
            'temp': prompt_form.temp.data,
        }
        print(f"api_key：{prompt_form.api_key.data} \ngpt_model{prompt_form.gpt_model.data} \ntemp{prompt_form.temp.data}")
        with open('settings.json', 'w') as f:
            json.dump(settings, f)
    
    if request.method == 'POST':
        # 返回一个响应给ajax请求表示保存成功
        return jsonify({'status': 'success'})
    else:
        # 渲染一个结果页面，并传递生成的提示词
        return render_template('index.html', prompt_form=prompt_form, question_form=question_form)



@app.route('/api/chat', methods=['POST'])
def chat():
    final_answer = " "
    question_form = QuestionForm()
    question = question_form.question.data
    print(f"开始提问↓↓↓")

    # FAISS向量数据库
    try:
        # 尝试通过FAISS文件夹路径加载向量数据库
        db = FAISS.load_local(FAISS_directory, embeddings)
    except Exception as e:
        # 本地向量数据库还没有内容引导进行内容添加
        error_message = f"【系统错误】本地向量数据库为空，请先添加学习资料！"
        print(f"本地向量数据库为空，请先添加学习资料！")
        return jsonify(answer = error_message)
        
    
    # 进行问答
    try:
        # 创建对话聊天记录总结
        summary_answer = sum_memory.load_memory_variables({})

        # 输出对话的摘要信息，并在向量数据库进行向量相似度查询
        # search_buff = summary_answer['chat_history'] + "\n新的问题是:" + question  
        #search_buff = question  
        #embedding_vector = embeddings.embed_query(search_buff)
        docs = db.similarity_search(question , k=10)


        # 移除每个匹配项中的特殊字符（如换行符和多余的空格），并按score顺序连接标题和内容，同时保证总token不超过8000
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
    
        # LLMChain启动
        final_answer = llm_chain.predict(combined_text = combined_text , question = question)

        # Return the answer as a JSON object
        html_answer = Markup(final_answer)
        print("final_answer",html_answer)
        return jsonify(answer = html_answer)
    except Exception as e:
        # 抛异常
        error_message =f"【系统错误】\n{e} \n ↑↑↑↑复制至ChatGPT可以查询解决方案"
        print(f"OpenAI异常")
        return jsonify(answer = error_message)



# 5. 获取可用端口
def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('',0))
    port = s.getsockname()[1]
    s.close()
    return port

def open_browser():
    time.sleep(1)  # 确保 Flask 服务器已启动
    webbrowser.open_new('http://127.0.0.1:{}'.format(port))

if __name__ == "__main__":
    port = find_free_port()
    print(f"Running on http://127.0.0.1:{port}")
    # 在一个独立的线程中打开浏览器，以防止阻止 Flask 服务器运行
    threading.Thread(target=open_browser).start()
    app.run(port=port)
