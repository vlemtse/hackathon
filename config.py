import os

bot_token = os.getenv('BOT_TOKEN')

gigachat_scope = os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS')
gigachat_auth_key = os.getenv('GIGACHAT_AUTH_KEY')
gigachat_model = 'GigaChat'

predict_model = 'rfr' # bert | rfr
