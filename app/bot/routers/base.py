from aiogram import Router
from aiogram.types import Message

from app.llm import get_llm_answer

from logging import getLogger


logger = getLogger(__name__)

router = Router()


@router.message()
async def echo(msg: Message):
    """
    Функция обрабатывает входящее сообщение, получает ответ от LLM и отправляет его обратно.
    """
    logger.info(f"Got message: {msg}")
    resp = await get_llm_answer(msg.text, msg.date)
    await msg.answer(resp)
