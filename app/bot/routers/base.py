from aiogram import Router
from aiogram.types import Message

from app.llm import get_llm_answer

from logging import getLogger


logger = getLogger(__name__)

router = Router()


@router.message()
async def echo(msg: Message):
    resp = await get_llm_answer(msg.text)
    await msg.answer(resp)
