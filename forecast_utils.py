import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----- LSTM imports -----
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------------------------------------------------
#  Feature engineering for Random Forest
# -------------------------------------------------------------------
def _build_features(y: pd.Series, lags=(1, 2, 3, 7, 14), rolls=(7, 14)):
    """
    Build lag and rolling-mean features for a univariate series y.
    Returns X (features) and y_clean (aligned target).
    """
    df = pd.DataFrame({"y": y})
    for l in lags:
        df[f"lag_{l}"] = df["y"].shift(l)
    for r in rolls:
        df[f"rollmean_{r}"] = df["y"].shift(1).rolling(r, min_periods=1).mean()

    # Day-of-week as simple seasonality proxy
    df["dow"] = df.index.dayofweek

    df = df.dropna()
    X = df.drop(columns=["y"])
    return X, df["y"]


# -------------------------------------------------------------------
#  Random Forest: recursive multi-step forecast
# -------------------------------------------------------------------
def _recursive_forecast_rf(y: pd.Series, horizon: int = 30):
    """
    Train a RandomForestRegressor on lag/rolling features and
    recursively forecast horizon days ahead.
    """
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

        next_date = last_date + pd.Timedelta(days=i + 1)
        y_ext.loc[next_date] = yhat
        future.append((next_date, yhat))

    f_dates = [d.strftime("%Y-%m-%d") for d, _ in future]
    f_vals = [round(float(v), 2) for _, v in future]
    return f_dates, f_vals, model


# -------------------------------------------------------------------
#  SARIMAX forecast
# -------------------------------------------------------------------
def _sarimax_forecast(y: pd.Series, horizon: int = 30):
    """
    Fit a simple SARIMAX model and forecast horizon days ahead.
    """
    order = (1, 1, 1)
    seasonal_order = (1, 0, 1, 7)

    try:
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        fcast = res.forecast(steps=horizon)
        f_dates = [d.strftime("%Y-%m-%d") for d in fcast.index]
        f_vals = [round(float(v), 2) for v in fcast.values]
    except Exception:
        # Fallback: simple last-value forecast
        last_val = float(y.iloc[-1])
        last_date = y.index[-1]
        f_dates = [
            (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(horizon)
        ]
        f_vals = [round(last_val, 2)] * horizon
        res = None

    return f_dates, f_vals, res


# -------------------------------------------------------------------
#  LSTM helper: train + multi-step forecast on a single series
# -------------------------------------------------------------------
def _lstm_forecast(y: pd.Series, horizon: int = 30, window: int = 14):
    """
    Train a small LSTM on the series y and recursively forecast horizon steps.
    """
    df = y.to_frame("y").copy()
    values = df["y"].values.astype(float)

    if len(values) <= window + 5:
        # Too little data – fall back to naive forecast
        last_val = float(values[-1])
        last_date = y.index[-1]
        f_dates = [
            (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(horizon)
        ]
        f_vals = [round(last_val, 2)] * horizon
        return f_dates, f_vals, None

    # Normalise to [0,1] for training stability
    min_val = values.min()
    max_val = values.max()
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scaled = (values - min_val) / denom

    # Build sequences
    X_list, Y_list = [], []
    for i in range(len(scaled) - window):
        X_list.append(scaled[i:i + window])
        Y_list.append(scaled[i + window])

    X = np.array(X_list)
    Y = np.array(Y_list)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(32, return_sequences=False, input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    cb = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X, Y, epochs=50, batch_size=8, verbose=0, callbacks=[cb])

    # Recursive forecast
    last_seq = scaled[-window:].reshape(1, window, 1)
    preds_scaled = []

    for _ in range(horizon):
        pred = model.predict(last_seq, verbose=0)[0][0]
        preds_scaled.append(pred)
        # Slide window
        last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)

    preds = [(p * denom) + min_val for p in preds_scaled]

    last_date = y.index[-1]
    f_dates = [
        (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(horizon)
    ]
    f_vals = [round(float(v), 2) for v in preds]
    return f_dates, f_vals, model


# -------------------------------------------------------------------
#  Backtest last k days (simple + robust)
# -------------------------------------------------------------------
def backtest_last_k(y: pd.Series, k: int = 14):
    """
    Simple, robust backtest on last k days.

    1) Split series into train + last k days as test.
    2) For each model (RF, SARIMAX, LSTM):
       - Fit on train
       - Forecast k steps ahead
       - Compute MAE, MAPE, RMSE, R2
       - Derive Accuracy (%) and Precision (%) from MAE / MAPE.
    """
    if len(y) < 25:  # safety for very short series
        return None

    k = min(k, len(y) // 3)
    if k <= 0:
        return None

    train = y.iloc[:-k]
    test = y.iloc[-k:]

    y_true = test.values.astype(float)

    # --- RF forecast on train ---
    try:
        rf_dates, rf_vals, _ = _recursive_forecast_rf(train, horizon=k)
        rf_pred = np.array(rf_vals, dtype=float)
    except Exception:
        rf_pred = np.full(k, np.nan)

    # --- SARIMAX forecast on train ---
    try:
        sx_dates, sx_vals, _ = _sarimax_forecast(train, horizon=k)
        sx_pred = np.array(sx_vals, dtype=float)
    except Exception:
        sx_pred = np.full(k, np.nan)

    # --- LSTM forecast on train ---
    try:
        lstm_dates, lstm_vals, _ = _lstm_forecast(train, horizon=k)
        lstm_pred = np.array(lstm_vals, dtype=float)
    except Exception:
        lstm_pred = np.full(k, np.nan)

    def _metrics(y_true_arr, y_pred_arr):
        mask = ~np.isnan(y_pred_arr)
        if mask.sum() == 0:
            return {
                "MAE": None,
                "MAPE": None,
                "RMSE": None,
                "R2": None,
                "ACCURACY": None,
                "PRECISION": None,
            }

        yt = y_true_arr[mask]
        yp = y_pred_arr[mask]

        mae = float(mean_absolute_error(yt, yp))
        mape = float(mean_absolute_percentage_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp))

        mean_true = float(np.mean(np.abs(yt)) + 1e-9)
        accuracy = 1.0 - mape                # fraction
        precision = 1.0 - (mae / mean_true)  # fraction

        return {
            "MAE": mae,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "ACCURACY": accuracy,
            "PRECISION": precision,
        }

    m_rf = _metrics(y_true, rf_pred)
    m_sx = _metrics(y_true, sx_pred)
    m_lstm = _metrics(y_true, lstm_pred)

    bt_dates = [d.strftime("%Y-%m-%d") for d in test.index]

    return {
        "k": int(k),
        "dates": bt_dates,
        "y_true": [float(v) for v in y_true],
        "rf_pred": [None if np.isnan(v) else float(v) for v in rf_pred],
        "sx_pred": [None if np.isnan(v) else float(v) for v in sx_pred],
        "lstm_pred": [None if np.isnan(v) else float(v) for v in lstm_pred],
        "metrics": {
            "RandomForest": m_rf,
            "SARIMAX": m_sx,
            "LSTM": m_lstm,
        },
    }


# -------------------------------------------------------------------
#  CSV loading helpers
# -------------------------------------------------------------------
def load_wide_csv(path: str):
    """
    Load the wide-format CSV:
        website, 2025-08-01, 2025-08-02, ...
    and return a DataFrame indexed by website with datetime columns.
    """
    df = pd.read_csv(path)
    if "website" not in df.columns:
        raise ValueError("CSV must have a 'website' column.")
    df = df.set_index("website")
    df.columns = pd.to_datetime(df.columns)
    return df


def get_series_for_site(wide_df: pd.DataFrame, site: str):
    """
    Extract a single site's time series (as a pd.Series) from the wide DF.
    """
    row = wide_df.loc[site]
    s = row.T
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s.astype(float)
    return s


# -------------------------------------------------------------------
#  Main API for app.py
# -------------------------------------------------------------------
def forecast_both(y: pd.Series, horizon: int = 30):
    """
    Return forecasts from RF + SARIMAX + LSTM.
    """
    rf_dates, rf_vals, _ = _recursive_forecast_rf(y, horizon=horizon)
    sx_dates, sx_vals, _ = _sarimax_forecast(y, horizon=horizon)
    lstm_dates, lstm_vals, _ = _lstm_forecast(y, horizon=horizon)

    return {
        "rf": {"dates": rf_dates, "values": rf_vals},
        "sx": {"dates": sx_dates, "values": sx_vals},
        "lstm": {"dates": lstm_dates, "values": lstm_vals},
    }
