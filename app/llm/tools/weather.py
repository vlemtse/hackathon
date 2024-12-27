from langchain.tools import tool
from model import predict_rfr_weather, predict_bert_weather, predict_lstm_weather

import config


@tool
def get_weather(hour:int, minuts:int, day: int, month: int, year: int) -> str:
    """
    Возвращает прогноз погоды на дату и время
    :param hour: час, принимает значение от 0 до 23 включительно
    :param minuts: минуты, принимает значение от 0 до 59 включительно
    :param day: день, принимает значение от 1 до 31 включительно
    :param month: месяц, принимает значение от 1 до 12 включительно
    :param year: год
    :return: температура на дату
    """

    if config.predict_model == "bert":
        resp = predict_bert_weather(f"{day}.{month}.{year}")
    elif config.predict_model == "lstm":
        resp = predict_lstm_weather(hour, minuts, day, month, year)
    else:
        resp = predict_rfr_weather(day, month, year)

    # Формируем ответ
    if config.predict_model == "bert":
        resp = (
            f"{resp}. Верни значение температуры вместе с точной датой прогноза и предложи варианты соответствующей одежды."
            f"так же верни количество осадков и облачность если они есть в контексте"
        )
    else:
        resp = (
                f"{resp}. Верни значение температуры вместе с точной датой прогноза, и предложи варианты соответствующей одежды."
        )
            
    return resp
