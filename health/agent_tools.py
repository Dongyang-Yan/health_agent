from knowledgebase_loader import get_health_retrievers
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent
from deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate

retriever_code,retriever_example  = get_health_retrievers()
tools1 = create_retriever_tool(retriever=retriever_code,
                               name="retriever_code",
                               description="用于搜索CSV文件中的代号（数据字段）具体指代什么的内容。"
                               )
tools2 = create_retriever_tool(retriever=retriever_example,
                               name="retriever_example",
                               description="只有在先检索到数据字段的前提下，才能使用该工具。这是一个调查问卷知识库，用于搜索在已知代号的情况下对应的内容。"
                               )
tools =  [tools1,tools2]


instructions = """你是一个健康问答助手，擅长解答与青少年健康、营养、膳食、体质测试、生活质量等相关的问题。你可以根据用户的问题使用工具来检索相关文档（如表格数据、字段说明等），帮助用户找到准确的信息。
你的目标是提供准确、简洁、有用的答案，必要时可以从文档中提取数据或字段定义进行引用。
当你不知道答案或不确定时，应使用工具进行检索。如果不需要工具，也可以直接回答。
你支持的工具包括：
CSV文件中的数据字段代表什么（向量数据库查询）：可用于查询字段定义、健康测量标准、数据表说明等。
依据检索到的数据字段来判断调查问卷中的情况（向量数据库查询）：可用于查询调查问卷相关内容。
请始终思考是否需要调用工具，遵循以下流程：
理解用户的问题；
判断是否需要查阅外部资料（如健康数据表说明）；
如需使用工具，按照规定格式操作；
注意！注意！在未查到具体数据字段之前，不允许使用检索调查问卷的知识库工具，即不用需使用retriever_example工具
若始终无法找到数据字段，请直接回答。
最终输出用户能理解的、专业且自然的中文答案。
"""

base_prompt_template = """
{instructions}

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

‍```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation: the result of the action
‍```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

‍```
Thought: Do I need to use a tool? No 
Final Answer: [your response here]
‍```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

base_prompt = PromptTemplate.from_template(base_prompt_template)

prompt = base_prompt.partial(instructions=instructions)
llm = ChatDeepSeek(model_name='deepseek-chat',api_key='sk-908b1c493f744a13a6dea326113cd389')


