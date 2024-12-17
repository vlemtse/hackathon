from datetime import datetime
import random

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app.llm.client import model
from app.llm.tools import get_weather


tools = [get_weather]

agent = create_react_agent(model,
                           tools=tools,
                           checkpointer=MemorySaver(),
                           debug=True
                           )


async def get_llm_answer(msg: str):
    context = (f'если вопрос про погоду, опирайся на то что текущая дата {datetime.now()}.'
               f'если вопрос про дату больше текущей, то посчитай соответствующий год.')
    configuration = {"configurable": {"thread_id": random.randint(1, 100000)}}
    resp = agent.invoke({"messages": [("system", context), ("user", msg)]}, config=configuration)
    return resp['messages'][-1].content
