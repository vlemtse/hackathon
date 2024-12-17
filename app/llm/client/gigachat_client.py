from langchain_gigachat.chat_models import GigaChat
import config


model = GigaChat(
    credentials=config.gigachat_auth_key,
    scope=config.gigachat_scope,
    model=config.gigachat_model,
    streaming=False,
    verify_ssl_certs=False,
)
