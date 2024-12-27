import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class TemperatureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TemperatureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_prediction_data(date_time, seq_length=24):
    date_time = pd.Timestamp(date_time)
    
    # Создаем признаки
    data = {
        'T': [0],  # будет заменено при предсказании
        'Зима': [1.0 if date_time.month in [12, 1, 2] else 0.0],
        'Весна': [1.0 if date_time.month in [3, 4, 5] else 0.0],
        'Лето': [1.0 if date_time.month in [6, 7, 8] else 0.0],
        'Осень': [1.0 if date_time.month in [9, 10, 11] else 0.0],
        'Январь': [1.0 if date_time.month == 1 else 0.0],
        'Февраль': [1.0 if date_time.month == 2 else 0.0],
        'Март': [1.0 if date_time.month == 3 else 0.0],
        'Апрель': [1.0 if date_time.month == 4 else 0.0],
        'Май': [1.0 if date_time.month == 5 else 0.0],
        'Июнь': [1.0 if date_time.month == 6 else 0.0],
        'Июль': [1.0 if date_time.month == 7 else 0.0],
        'Август': [1.0 if date_time.month == 8 else 0.0],
        'Сентябрь': [1.0 if date_time.month == 9 else 0.0],
        'Октябрь': [1.0 if date_time.month == 10 else 0.0],
        'Ноябрь': [1.0 if date_time.month == 11 else 0.0],
        'Декабрь': [1.0 if date_time.month == 12 else 0.0],
        'Светло': [1.0 if (
            (date_time.month in [12, 1, 2] and 8 <= date_time.hour <= 16) or
            (date_time.month in [3, 4, 5] and 6 <= date_time.hour <= 19) or
            (date_time.month in [6, 7, 8] and 4 <= date_time.hour <= 22) or
            (date_time.month in [9, 10, 11] and 7 <= date_time.hour <= 17)
        ) else 0.0]
    }
    
    # Преобразуем в DataFrame и затем в тензор
    df = pd.DataFrame(data)
    features = df.values
    features = np.tile(features, (seq_length, 1))
    features = torch.FloatTensor(features).unsqueeze(0)
    
    return features

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TemperatureLSTM(
        input_size=18,
        hidden_size=64,
        num_layers=2
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def predict_temperature(date_time, model):
    device = next(model.parameters()).device
    
    with torch.no_grad():
        prepared_data = prepare_prediction_data(date_time)
        prepared_data = prepared_data.to(device)
        prediction = model(prepared_data)
        return prediction.item()
