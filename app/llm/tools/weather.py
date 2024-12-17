from langchain.tools import tool
from model import predict_rfr_weather, predict_bert_weather

import config


@tool
def get_weather(day: int, month: int, year: int) -> str:
    """
    Возвращает прогноз погоды на дату
    :param day: день, принимает значение от 1 до 31 включительно
    :param month: месяц, принимает значение от 1 до 12 включительно
    :param year: год
    :return: температура на дату
    """

    if config.predict_model == 'bert':
        resp = predict_bert_weather(f'{day}.{month}.{year}')
    else:
        resp = predict_rfr_weather(day, month, year)

    # Формируем ответ
    resp = f'{resp}. Верни значение температуры и предложи варианты соответствующей одежды.'
    return resp
