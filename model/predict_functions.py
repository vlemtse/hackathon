import os
import joblib
import torch
import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification


current_directory = os.getcwd()


def predict_bert_weather(date: str) -> str:
    base_dir = f"{current_directory}/model/bert_weather_model"
    scaler = joblib.load(f'{base_dir}/scaler.pkl')

    # Загрузка токенизатора
    tokenizer = BertTokenizer.from_pretrained(base_dir)

    # Загрузка модели
    model = BertForSequenceClassification.from_pretrained(base_dir)

    # Переносим модель на устройство (GPU или CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode_plus(
            date,
            add_special_tokens=True,
            max_length=10,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = scaler.inverse_transform(outputs.logits.cpu().numpy())

    res = (f'Температура: {predictions[0][0]}, '
           f'количество осадков в мм: {predictions[0][1]}, '
           f'облачность: {predictions[0][2]}%')
    return res


def predict_rfr_weather(day: int, month: int, year: int) -> str:
    # Загружаем сохранённую модель
    loaded_model = joblib.load(f"{current_directory}/model/random_forest_model/random_forest_model.joblib")
    pred_d = pd.DataFrame(
        {
            'day': [day],
            'month': [month],
            'year': [year],
        }
    )
    loaded_model.predict(pred_d)
    # Делаем предсказание
    temperature = round(loaded_model.predict(pred_d)[0], 0)

    res = f'Температура: {temperature}'
    return res
