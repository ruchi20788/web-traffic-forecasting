import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

def _build_features(y: pd.Series, lags=(1,2,3,7,14), rolls=(7,14)):
    df = pd.DataFrame({"y": y})
    for l in lags:
        df[f"lag_{l}"] = df["y"].shift(l)
    for r in rolls:
        df[f"rollmean_{r}"] = df["y"].shift(1).rolling(r, min_periods=1).mean()
    df["dow"] = df.index.dayofweek
    df = df.dropna()
    X = df.drop(columns=["y"])
    return X, df["y"]

def _recursive_forecast_rf(y: pd.Series, horizon=30):
    X, y_train = _build_features(y)
    if len(X) < 10:
        raise ValueError("Not enough data for RandomForest features.")
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X, y_train)

    future = []
    y_ext = y.copy()
    last_date = y_ext.index[-1]

    for i in range(horizon):
        Xf, _ = _build_features(y_ext)
        x_next = Xf.iloc[[-1]]
        yhat = float(model.predict(x_next)[0])
        if yhat < 0:
            yhat = 0.0
        next_date = last_date + pd.Timedelta(days=i+1)
        y_ext.loc[next_date] = yhat
        future.append((next_date, yhat))

    f_dates = [d.strftime("%Y-%m-%d") for d, _ in future]
    f_vals = [round(float(v), 2) for _, v in future]
    return f_dates, f_vals, model

def _sarimax_forecast(y: pd.Series, horizon=30):
    order = (1,1,1)
    seasonal_order = (1,0,1,7)
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fcast = res.forecast(steps=horizon)
    f_dates = [d.strftime("%Y-%m-%d") for d in fcast.index]
    f_vals = [round(float(v), 2) for v in fcast.values]
    return f_dates, f_vals, res

def backtest_last_k(y: pd.Series, k=14):
    k = min(k, max(7, len(y)//4))
    split = len(y) - k
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    preds_rf = []
    y_rf_series = y_train.copy()
    for i in range(k):
        try:
            X, ytr = _build_features(y_rf_series)
            if len(X) < 10:
                preds_rf = [np.nan]*k
                break
            rf = RandomForestRegressor(n_estimators=300, random_state=42)
            rf.fit(X, ytr)
            Xf, _ = _build_features(y_rf_series)
            yhat = float(rf.predict(Xf.iloc[[-1]])[0])
            preds_rf.append(max(0.0, yhat))
        except Exception:
            preds_rf.append(np.nan)
        y_rf_series = pd.concat([y_rf_series, y_test.iloc[[i]]])

    preds_sx = []
    y_sx_series = y_train.copy()
    for i in range(k):
        try:
            sx = SARIMAX(y_sx_series, order=(1,1,1), seasonal_order=(1,0,1,7), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            yhat = float(sx.forecast(steps=1)[0])
            preds_sx.append(max(0.0, yhat))
        except Exception:
            preds_sx.append(np.nan)
        y_sx_series = pd.concat([y_sx_series, y_test.iloc[[i]]])

    y_true = y_test.values
    def _metrics(y_true, y_pred):
        import numpy as _np
        mask = ~_np.isnan(y_pred)
        if mask.sum() == 0:
            return {"MAPE": None, "RMSE": None}
        mape = float(mean_absolute_percentage_error(y_true[mask], _np.array(y_pred)[mask]))
        rmse = float(np.sqrt(mean_squared_error(y_true[mask], _np.array(y_pred)[mask])))
        return {"MAPE": mape, "RMSE": rmse}

    m_rf = _metrics(y_true, np.array(preds_rf))
    m_sx = _metrics(y_true, np.array(preds_sx))

    return {
        "k": int(k),
        "dates": [d.strftime("%Y-%m-%d") for d in y.iloc[-k:].index],
        "y_true": [int(v) for v in y.iloc[-k:].values],
        "rf_pred": [None if np.isnan(v) else round(float(v),2) for v in preds_rf],
        "sx_pred": [None if np.isnan(v) else round(float(v),2) for v in preds_sx],
        "metrics": {
            "RandomForest": m_rf,
            "SARIMAX": m_sx
        }
    }

def load_wide_csv(path: str):
    df = pd.read_csv(path)
    if "website" not in df.columns:
        raise ValueError("CSV must have a 'website' column.")
    df = df.set_index("website")
    df.columns = pd.to_datetime(df.columns)
    return df

def get_series_for_site(wide_df: pd.DataFrame, site: str):
    row = wide_df.loc[site]
    s = row.T
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s.astype(float)
    return s

def forecast_both(y: pd.Series, horizon=30):
    rf_dates, rf_vals, _ = _recursive_forecast_rf(y, horizon=horizon)
    sx_dates, sx_vals, _ = _sarimax_forecast(y, horizon=horizon)
    return {
        "rf": {"dates": rf_dates, "values": rf_vals},
        "sx": {"dates": sx_dates, "values": sx_vals}
    }