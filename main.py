from asyncio import run
from app import bot_start
from logsett import setup_logging


if __name__ == "__main__":
    # Настройка логирования
    setup_logging()

    # Запуск бота
    run(bot_start())
