import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib, os

BASE = os.path.dirname(__file__)
csv_path = os.path.join(BASE, 'data.csv')
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

for col in ['Open','High','Low','Close']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','').str.strip(), errors='coerce')

df['Target'] = df['Close'].shift(-1)
df = df.dropna(subset=['Open','High','Low','Close','Target']).reset_index(drop=True)

X = df[['Open','High','Low','Close']].values
y = df['Target'].values

split_idx = int(len(X)*0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)

preds = model.predict(X_test_s)
rmse = np.sqrt(np.mean((y_test - preds)**2))
print("Training complete. Test RMSE:", rmse)

joblib.dump(model, os.path.join(BASE, 'linear_reg_model.pkl'))
joblib.dump(scaler, os.path.join(BASE, 'scaler.pkl'))
print("Saved linear_reg_model.pkl and scaler.pkl")
