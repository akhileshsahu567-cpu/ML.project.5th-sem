from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib, os
from typing import Optional
import pandas as pd

app = FastAPI(title='Nifty50 Predictor API (FastAPI)')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_reg_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
DATA_CSV = os.path.join(os.path.dirname(__file__), 'data.csv')

model = None
scaler = None

# Try to load model & scaler if present
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
except Exception as e:
    model = None
    scaler = None

class OHLC(BaseModel):
    open: float
    high: float
    low: float
    close: float

def mock_predict(close: float, history_closes=None):
    # Simple mock prediction: momentum + small noise
    if history_closes and len(history_closes) > 0:
        momentum = close - history_closes[-1]
        vol = max(1.0, (max(history_closes) - min(history_closes)) / max(1, len(history_closes)))
    else:
        momentum = 0.0
        vol = max(20.0, close * 0.002)
    noise = (np.random.rand() - 0.5) * vol * 2
    pred = close + 0.4 * momentum + noise
    return float(np.round(pred, 2))

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}

@app.post('/predict')
def predict(payload: OHLC):
    x = np.array([[payload.open, payload.high, payload.low, payload.close]], dtype=float)
    # If model available, use it
    if model is not None and scaler is not None:
        try:
            x_scaled = scaler.transform(x)
            p = model.predict(x_scaled)[0]
            return {'predicted_next_close': float(np.round(p, 2)), 'model': True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Model prediction failed: {e}')
    # else mock predict (optionally use data.csv if present)
    history_closes = None
    try:
        if os.path.exists(DATA_CSV):
            df = pd.read_csv(DATA_CSV)
            if 'Close' in df.columns:
                history_closes = df['Close'].dropna().astype(float).tolist()[-50:]
    except Exception:
        history_closes = None
    pred = mock_predict(payload.close, history_closes)
    return {'predicted_next_close': pred, 'model': False}

@app.get('/history')
def history(limit: Optional[int] = 200):
    # If data.csv exists and model/scaler present, compute test-set preds similar to training split
    if os.path.exists(DATA_CSV):
        try:
            df = pd.read_csv(DATA_CSV)
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.sort_values('Date').reset_index(drop=True)
            for col in ['Open','High','Low','Close']:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','').str.strip(), errors='coerce')
            df['Target'] = df['Close'].shift(-1)
            df = df.dropna(subset=['Open','High','Low','Close','Target']).reset_index(drop=True)
            X = df[['Open','High','Low','Close']].copy()
            y = df['Target'].copy()
            split_idx = int(len(X) * 0.8)
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
            # if model exists, predict
            preds = None
            if model is not None and scaler is not None:
                preds = model.predict(scaler.transform(X_test))
            rows = []
            for i in range(min(len(X_test), limit)):
                rows.append({
                    'date': df['Date'].iloc[split_idx + i].strftime('%Y-%m-%d'),
                    'open': float(X_test['Open'].iloc[i]),
                    'high': float(X_test['High'].iloc[i]),
                    'low': float(X_test['Low'].iloc[i]),
                    'close': float(X_test['Close'].iloc[i]),
                    'actual': float(y_test.iloc[i]),
                    'predicted': float(np.round(preds[i],2)) if preds is not None else None
                })
            metrics = {}
            if preds is not None:
                metrics['rmse'] = float(np.round(np.sqrt(np.mean((y_test.values - preds)**2)),4))
                metrics['n_test'] = int(len(y_test))
            return {'history': rows, 'metrics': metrics}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to prepare history: {e}')
    else:
        return {'history': [], 'metrics': {}}
