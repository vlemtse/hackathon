from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from .routers import base_router
import config

from logging import getLogger


logger = getLogger(__name__)

dp = Dispatcher(storage=MemoryStorage())


async def bot_start():
    bot = Bot(token=config.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # Добавлять роутеры тут:
    dp.include_router(base_router)

    # Удаляет/игнорирует все сообщения, которые были написаны пока бот не работал
    await bot.delete_webhook(drop_pending_updates=True)

    # Запуск бота
    logger.info("Bot starting...")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
