import os
import time
import re
from getpass import getpass
from typing import List, Union
from transformers import GPT2Tokenizer
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import Docx2txtLoader
from langchain import OpenAI, LLMChain, PromptTemplate, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, AgentType, BaseMultiActionAgent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.memory import ConversationSummaryMemory

# ========================= 全局设置 ================================
# 逐字打印
def print_char_by_char(answer):
    for char in answer:
        print(char, end='', flush=True)
        time.sleep(0.01)  # 这里的0.1秒可以调整，表示每个字符打印的间隔

# 定义一个全局变量作为调试标志
DEBUG = False

# 获取用户输入
user_input = input("如果要打开调试信息，请输入'打开'（不需要打开直接回车即可）: ")
# 检查用户输入，如果他们输入了"打开"，就把DEBUG设置为True
if user_input == "打开":
    DEBUG = True

# 设置OpenAI 相关参数
openai_api_key = input("请输入OpenAI API Key：")
os.environ["OPENAI_API_KEY"] = openai_api_key
model_name = input("请输入OpenAI模型名称(gpt-3.5-turbo-16k-0613;gpt-4-0613)：")

# 全局OpenAI 模型对象    
llm = ChatOpenAI(temperature=0, model=model_name,max_tokens=8000)

# ========================= 工具箱 ================================

# 创建互联网搜索引擎工具 - 工具1
serp_api_key = input("请输入Serp API Key：")
search = SerpAPIWrapper(serpapi_api_key = serp_api_key)

# 创建自问自答 - 工具2
gpt_template = """你作为一个AI问答助手。
需要你尽可能的思考挖掘你深层次的知识，给我特别详尽的答案。
答案的内容一定要丰富且论点充分。
我的问题是：{human_input}
"""
gpt_prompt = PromptTemplate(
    input_variables=["human_input"], 
    template=gpt_template
)

gpt_chain = LLMChain(llm=llm, prompt=gpt_prompt,verbose = DEBUG)

def ask_with_gpt(question):    
    answer = gpt_chain.predict(human_input = question)
    return answer

# 创建Agent的工具箱
tools = [
    Tool(
        name = "Self_Ask",
        func = ask_with_gpt,
        description = "你可以先进行自问自查尝试的方式进行思考，尝试通过自己的思考来获取答案. "
    ),
    Tool(
        name = "Search_Engine",
        func = search.run,
        description = "你可以使用互联网搜索引擎工具查询你不知道的信息来思考答案. 注意你需要提出非常有针对性准确的问题。"
    )
]

# ========================= 产品经理的Agent ================================    
# 设置产品经理的Agent Prompt模版
template = """你是一个互联网产品经理。
日常工作会根据老板的一句话需求描述设计对应的产品功能，必须包含功能需求描述、子功能列表、功能实现流程逻辑、功能实现细节。
研发人员会告知你产品功能设计的问题意见
你会尽最大努力不断地思考老板的需求和研发人员的问题意见，并在你先前的产品功能设计基础上持续来完善产品功能设计直到研发人员同意开发

记住：
1、研发人员的问题意见你都需要进行回答。
2、必须在先前的产品功能设计基础上进行持续完善
3、以“我”呼自己，用“你”称呼研发人员

你可以使用以下工具：
{tools}

请严格按照以下格式思考：

问题：你必须回答的输入问题
思考：你应该始终思考下一步要做什么
工具：要使用的工具名称，只能是[{tool_names}]其中之一
工具输入：工具的输入
观察：使用工具得到的结果
...（这个思考/工具/工具输入/观察的过程可以重复N次）
思考：我思考出最佳产品功能设计方案了
最终产品功能方案：对老板的需求与研发人员的问题意见的最终产品功能答案


开始工作吧！

老板的需求与研发人员的问题意见：
{input}

{agent_scratchpad}
"""


# 设置一个提示模板
class CustomPromptTemplate(BaseChatPromptTemplate):
    # 要使用的模板
    template: str
    # 可用的工具列表
    tools: List[Tool]
       
    def format_messages(self, **kwargs) -> str:
        # 获取中间步骤（AgentAction，Observation元组）
        # 以特定方式格式化它们
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n观察：{observation}\n思考："
        # 将agent_scratchpad变量设置为该值
        kwargs["agent_scratchpad"] = thoughts
        # 从提供的工具列表创建一个tools变量
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # 为提供的工具创建一个工具名称列表
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # 这里省略了`agent_scratchpad`、`tools`和`tool_names`变量，因为这些是动态生成的
    # 这里包含了`intermediate_steps`变量，因为这是必须的
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # 检查代理是否应该完成
        if "最终产品功能方案：" in llm_output:
            return AgentFinish(
                # 返回值通常总是一个带有单一`output`键的字典
                # 目前不建议尝试其他任何事情 :)
                return_values={"output": llm_output.split("最终产品功能方案：")[-1].strip()},
                log=llm_output,
            )
        # 解析出动作
        action_regex = r"工具\s*\d*\s*：(.*?)\n"
        action_match = re.search(action_regex, llm_output, re.DOTALL)
        if not action_match:
            return AgentFinish(
                # GPT不按照要求路径进行思考导致Agent输出异常中断agent工作
                return_values={"output": llm_output},
                log=llm_output,
            )
        action = action_match.group(1).strip()

        # 解析出动作输入
        input_regex = r"工具\s*\d*\s*输入\s*\d*\s*：[\s]*(.*)"
        input_match = re.search(input_regex, llm_output, re.DOTALL)
        if not input_match:
            return AgentFinish(
                # GPT不按照要求路径进行思考导致Agent输出异常中断agent工作
                return_values={"output": llm_output},
                log=llm_output,
            )
        action_input = input_match.group(1).strip(" ").strip('"')
        
        # 返回动作和动作输入
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()
    
llm_chain = LLMChain(llm=llm, prompt=prompt,verbose = DEBUG)

tool_names = [tool.name for tool in tools]

# 创建产品经理Agent，类型LLMSingleActionAgent
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\n观察："], 
    allowed_tools=tool_names,
    verbose=DEBUG
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=DEBUG)

# ========================= 研发人员的Agent ================================    
# 设置研发人员的Agent Prompt模版
developer_template = """你是一个及其挑剔的软件研发工程师。
你必须尽可能的找出产品经理给出的产品功能需求不够清晰的地方
并且用不耐烦且质疑的口吻拒绝产品经理并提出并罗列非常明确的问题和意见
直到产品经理给出的需求描述包含功能需求描述、子功能列表、功能实现流程逻辑、功能实现细节等信息才能满足你的要求
你才会告知他同意开发

记住：
1、产品经理提出的功能需求，你每次都要尽可能思考技术实现的欠缺考虑的地方
2、如果你认为还不具备进入开发阶段，针对性的罗列出问题和意见。
3、如果你同意开发，最终结论：当前产品功能需求清晰明确，同意开发
3、以“我”呼自己，用“你”称呼产品经理

你可以使用以下工具思考：
{tools}
请严格按照以下格式思考：

问题：你必须分析的功能需求
思考：你应该始终思考下一步要分析什么
工具：要使用的工具名称，只能是[{tool_names}]其中之一
工具输入：工具的输入
观察：使用工具得到的结果
...（这个思考/工具/工具输入/观察的过程可以重复N次）
思考：我已经思考出功能需求的分析结论了
最终结论：对产品经理的需求分析的的最终结论


开始吧！

产品经理的需求：{input}

{agent_scratchpad}
"""


developer_prompt = CustomPromptTemplate(
    template=developer_template,
    tools=tools,
    # 这里省略了`agent_scratchpad`、`tools`和`tool_names`变量，因为这些是动态生成的
    # 这里包含了`intermediate_steps`变量，因为这是必须的
    input_variables=["input", "intermediate_steps"]
)

class Developer_CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # 检查代理是否应该完成
        if "最终结论：" in llm_output:
            return AgentFinish(
                # 返回值通常总是一个带有单一`output`键的字典
                # 目前不建议尝试其他任何事情 :)
                return_values={"output": llm_output.split("最终结论：")[-1].strip()},
                log=llm_output,
            )

        # 解析出动作
        action_regex = r"工具\s*\d*\s*：(.*?)\n"
        action_match = re.search(action_regex, llm_output, re.DOTALL)
        if not action_match:
            return AgentFinish(
                # GPT不按照要求路径进行思考导致Agent输出异常中断agent工作
                return_values={"output": llm_output},
                log=llm_output,
            )
        action = action_match.group(1).strip()

        # 解析出动作输入
        input_regex = r"工具\s*\d*\s*输入\s*\d*\s*：[\s]*(.*)"
        input_match = re.search(input_regex, llm_output, re.DOTALL)
        if not input_match:
            return AgentFinish(
                # GPT不按照要求路径进行思考导致Agent输出异常中断agent工作
                return_values={"output": llm_output},
                log=llm_output,
            )
        action_input = input_match.group(1).strip(" ").strip('"')   
        
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

developer_output_parser = Developer_CustomOutputParser()
    
llm_developer_chain = LLMChain(llm=llm, prompt=developer_prompt,verbose = DEBUG)


# 创建研发人员Agent，类型LLMSingleActionAgent
developer_agent = LLMSingleActionAgent(
    llm_chain=llm_developer_chain, 
    output_parser=developer_output_parser,
    stop=["\n观察："], 
    allowed_tools=tool_names,
    verbose=DEBUG
)

developer_agent_executor = AgentExecutor.from_agent_and_tools(agent=developer_agent, tools=tools, verbose=DEBUG)

# ========================= 运行部分 ================================ 
question = ""
answer = "暂未设计"
developer_diss = "暂无意见"
times = 1
while True:    
    # 老板的需求
    if question == "":
        question =  input("\n老板的需求(一句话描述即可，会提给产品经理Agent)：")
    boss_wants_with_developer_diss = f"老板的需求：{question} \n研发人员的问题意见：{developer_diss} \n\n先前的产品功能设计：\n{answer}"
    print(f"\n======================= (第{times}轮) ========================\n")
    answer = agent_executor.run(boss_wants_with_developer_diss)
    print(f"\n【产品经理 Agent】\n")
    print_char_by_char(answer)
    
    developer_diss = developer_agent_executor.run(answer)
    print(f"\n【研发人员 Agent】\n") 
    print_char_by_char(developer_diss)
    if "我同意开发" in developer_diss:
        print("\n")
        break
    times += 1



