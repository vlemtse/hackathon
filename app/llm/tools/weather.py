from langchain.tools import tool


@tool
def get_weather(day: int, month: int, year: int) -> str:
    """
    Возвращает прогноз погоды на дату
    :param day: день, принимает значение от 1 до 31 включительно
    :param month: месяц, принимает значение от 1 до 12 включительно
    :param year: год
    :return: температура на дату
    """

    #TODO тут обращение к модели прогнозирования
    temperature = -11.5 # Мок
    ret = f'Ожидаемая температура {temperature}. Предложи варианты соответствующей одежды.'
    return ret
