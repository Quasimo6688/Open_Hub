import tkinter as tk
from tkinter import scrolledtext, messagebox
from langchain.llms import OpenAI
import os
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
import time

# 设置OpenAI API密钥（请谨慎处理这一部分，以防不必要的安全风险）
try:
    openai_api_key = "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"
    os.environ["OPENAI_API_KEY"] = openai_api_key
except Exception as e:
    messagebox.showerror("错误", f"无法设置API密钥: {e}")

# 创建记忆实例
memory = ConversationBufferMemory(memory_key="chat_history")

# Prompt模版
template = """你是一个AI智能问答助手。
"{chat_history}"
现在我的问题是: {human_input}
"""

# 创建Prompt模版
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)

# 创建LLMChain
try:
    llm_chain = LLMChain(
        llm=OpenAI(temperature=0, model_name="text-davinci-003"),
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
except Exception as e:
    messagebox.showerror("错误", f"无法初始化LLMChain: {e}")

# 创建主窗口
window = tk.Tk()
window.title("Advanced ChatBot")
window.geometry("800x600")
window.configure(bg="#2C3E50")

# 创建滚动文本框
chat_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=70, height=30, bg="#34495E", fg="white", font=("Arial", 12))
chat_text.grid(row=0, column=0, padx=20, pady=20, columnspan=4)

# 创建输入框
input_entry = tk.Entry(window, width=60, bg="#7F8C8D", fg="white", font=("Arial", 12))
input_entry.grid(row=1, column=0, padx=20, pady=10, columnspan=3)

# 发送按钮
def send_message():
    user_input = input_entry.get()
    chat_text.insert(tk.END, f"You: {user_input}\n")
    input_entry.delete(0, tk.END)
    try:
        answer = llm_chain.predict(human_input=user_input)
        chat_text.insert(tk.END, f"Bot: {answer}\n")
    except Exception as e:
        messagebox.showerror("Error", f"Could not get an answer: {e}")

send_button = tk.Button(window, text="Send", command=send_message, bg="#3498DB", fg="white", font=("Arial", 12))
send_button.grid(row=1, column=3, padx=20, pady=10)

if __name__ == "__main__":
    window.mainloop()
