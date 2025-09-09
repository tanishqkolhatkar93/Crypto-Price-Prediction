# Crypto Price Prediction 🚀

This project implements a deep learning-based system to predict Bitcoin prices using historical data fetched from the [CryptoCompare API](https://min-api.cryptocompare.com/). The model leverages **LSTM (Long Short-Term Memory)** neural networks to capture temporal dependencies in time-series data.

## 📌 Features
- Fetches real-time Bitcoin historical price data using the CryptoCompare API
- Data preprocessing (cleaning, normalization, and sequence generation)
- LSTM-based deep learning model for time-series forecasting
- Model evaluation using **Mean Absolute Error (MAE)**
- Visualization of predicted vs actual Bitcoin prices

## 📊 Dataset
- Source: [CryptoCompare API](https://min-api.cryptocompare.com/)
- Cryptocurrency: **Bitcoin (BTC)**
- Currency: **CAD**
- 500 days of historical daily data

## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **CryptoCompare API**

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/tanishqkolhatkar93/Crypto-Price-Prediction.git
cd Crypto-Price-Prediction
pip install -r requirements.txt
```

## ▶️ Usage
Run the notebook in **Google Colab** or locally:

```python
import json
import requests
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
```

Fetch historical Bitcoin data:
```python
endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=CAD&limit=500')
hist = pd.DataFrame(json.loads(res.content)['Data'])
hist = hist.set_index('time')
hist.index = pd.to_datetime(hist.index, unit='s')
target_col = 'close'
```

Train and evaluate the LSTM model, then visualize predictions.

## 📈 Results
- The LSTM model achieved **promising accuracy** on Bitcoin price prediction.
- Example visualization:

Predicted vs Actual Prices 📊

## 🔮 Future Work
- Extend to other cryptocurrencies (ETH, LTC, etc.)
- Deploy as a real-time price prediction API
- Experiment with GRU and Bi-LSTM models

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

## 📜 License
This project is licensed under the MIT License.

---
👨‍💻 Developed by [Tanishq Kolhatkar](https://github.com/tanishqkolhatkar93)
