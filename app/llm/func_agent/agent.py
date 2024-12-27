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


async def get_llm_answer(msg: str, date_time:datetime):
    # context = (f'если вопрос про погоду, опирайся на то что текущая дата {date_time}.'
    #            f'если вопрос про дату больше текущей, то посчитай соответствующий год.')
    context = (
               f"Ты — ИИ-ассистент, который отвечает на вопросы о погоде."
               f" Текущая дата и время: {date_time}. "
               f"Если пользователь спрашивает про погоду, определи дату или время, на которое он хочет узнать прогноз."
               f" Если дата указана как 'завтра', 'послезавтра' или 'через N дней', рассчитай её, используй время : {date_time.hour} - часы,  {date_time.minute} - минуты."
               f'если вопрос про дату больше текущей, то посчитай соответствующий год.'
               )
    configuration = {"configurable": {"thread_id": random.randint(1, 100000)}}
    resp = agent.invoke({"messages": [("system", context), ("user", f'{msg} . Время вопроса {date_time}')]}, config=configuration)
    return resp['messages'][-1].content
