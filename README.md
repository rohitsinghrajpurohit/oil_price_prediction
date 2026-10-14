# Oil Price Prediction (Python + Flask)

Dataset: `data/Oil_Data.csv` (daily oil price: Date, Price, Open, High, Low, Volume, Chg%).

## Run
```bash
pip install -r requirements.txt
python prepare_data.py   # raw CSV clean karta hai
python train.py          # models train + compare + save
python flask_app.py      # http://127.0.0.1:5000
```

## Pipeline
1. **prepare_data.py**: blank rows hatao, date parse, Volume (K) number me, Volume=0 ko missing maano.
2. **features.py**: pichle 5 closing prices se features (4 daily returns, 3/5-day average ratio, volatility). Target = agle din ka return.
3. **train.py**: Ridge, Random Forest, Gradient Boosting. Time-based 80/20 split, TimeSeriesSplit CV se best model, naive baseline (kal = aaj) se comparison. Final model poore data pe refit.
4. **flask_app.py + templates/index.html**: history chart, N-day forecast (95% range ke saath), custom 5-price prediction, model comparison, actual vs predicted.

## Limitations
- Sirf 210 trading days (Jan-Oct 2022), isliye results noisy hain.
- Koi model naive baseline ko clearly beat nahi karta. Oil price random-walk jaisa hota hai.
- Multi-day forecast recursive hai, to door ke din ki uncertainty badhti hai.
