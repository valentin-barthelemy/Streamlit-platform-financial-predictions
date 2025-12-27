import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple

# Stats / ML de base
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# --- TensorFlow (compat macOS / Apple Silicon) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    try:
        for dev in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(dev, True)
    except Exception:
        pass
except Exception:
    TF_AVAILABLE = False

# XGBoost (optionnel)
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# =========================
# Erreur UX (pour l'app)
# =========================
class UXError(Exception):
    def __init__(self, kind: str, msg: str, details: str = ""):
        super().__init__(msg)
        self.kind = kind
        self.details = details


# =========================
# Données & Prétraitements
# =========================
def load_prices(ticker: str, period="3y", interval="1d") -> pd.DataFrame:
    """Télécharge prix mono-ticker et normalise les colonnes."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise UXError("data", "Aucune donnée téléchargée pour ce ticker/période.")

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = pd.DataFrame({
                "Open":  df["Open"][ticker],
                "High":  df["High"][ticker],
                "Low":   df["Low"][ticker],
                "Close": df["Close"][ticker],
                "Adj Close": df["Adj Close"][ticker] if ("Adj Close" in df.columns.levels[0]) else df["Close"][ticker],
                "Volume": df["Volume"][ticker]
            })
        except Exception:
            df = df.xs(df.columns.levels[1][0], axis=1, level=1)
            df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]

    df = df.rename(columns=str.title)
    return df


def get_company_name(ticker: str) -> str:
    try:
        t = yf.Ticker(ticker)
        info = t.get_info()
        name = info.get("longName") or info.get("shortName") or ticker
        return f"{name} ({ticker})"
    except Exception:
        return ticker


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - 100 / (1 + rs)


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    close, log_ret, vol20/vol60, rsi, ma5, ma20, mom5 (close/ma20-1), rv20 (vol réalisée 20j).
    """
    close = prices["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)

    log_ret = np.log(close).diff().rename("log_ret")
    vol20 = log_ret.rolling(20).std().rename("vol20")
    vol60 = log_ret.rolling(60).std().rename("vol60")
    rsi = _rsi(close, 14).rename("rsi")
    ma5 = close.rolling(5).mean().rename("ma5")
    ma20 = close.rolling(20).mean().rename("ma20")
    mom5 = (close / (ma20 + 1e-12) - 1.0).rename("mom5")
    rv20 = np.sqrt((log_ret ** 2).rolling(20).sum()).rename("rv20")

    feat = pd.concat([close.rename("close"), log_ret, vol20, vol60, rsi, ma5, ma20, mom5, rv20], axis=1).dropna()
    return feat


# =========================
# Aides ML
# =========================
def _build_ml_matrix(feat: pd.DataFrame, max_lag: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """X = lags 1..max_lag + exog lissées; y = log_ret(t)."""
    df = feat.copy()
    for k in range(1, max_lag + 1):
        df[f"lag{k}"] = df["log_ret"].shift(k)

    df["vol20_l1"] = df["vol20"].shift(1)
    df["vol60_l1"] = df["vol60"].shift(1)
    df["rsi_l1"] = df["rsi"].shift(1)
    df["mom5_l1"] = df["mom5"].shift(1)
    df["rv20_l1"] = df["rv20"].shift(1)

    cols = [c for c in df.columns if c.startswith("lag")] + ["vol20_l1", "vol60_l1", "rsi_l1", "mom5_l1", "rv20_l1"]
    X = df[cols].dropna()
    y = df.loc[X.index, "log_ret"]
    return X, y


# =========================
# Modèles régression & Backtest
# =========================
def sarima_walk_and_forecast(log_ret: pd.Series, test_days: int = 30, horizon: int = 5, order=(1, 1, 1)) -> Dict:
    ret = log_ret.dropna()
    dates_test = pd.DatetimeIndex(ret.index[-test_days:]) if len(ret) >= test_days else ret.index
    preds = []

    for d in dates_test:
        train = ret[ret.index < d]
        if len(train) < 60:
            preds.append(np.nan)
            continue
        model = SARIMAX(train, order=order, seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        preds.append(float(res.get_forecast(steps=1).predicted_mean.iloc[0]))

    model_full = SARIMAX(ret, order=order, seasonal_order=(0, 0, 0, 0),
                         enforce_stationarity=False, enforce_invertibility=False)
    res_full = model_full.fit(disp=False)
    fc5 = res_full.get_forecast(steps=horizon).predicted_mean.values
    return {"name": f"SARIMA{order}", "test_dates": dates_test, "test_pred": np.array(preds), "fc5": fc5}


def linreg_walk_and_forecast(feat: pd.DataFrame, test_days: int = 30, horizon: int = 5, max_lag: int = 10) -> Dict:
    X, y = _build_ml_matrix(feat, max_lag=max_lag)
    idx = X.index
    dates_test = pd.DatetimeIndex(idx[-test_days:]) if len(idx) >= test_days else idx
    preds = []

    for d in dates_test:
        mask = idx < d
        Xtr, ytr = X.loc[mask], y.loc[mask]
        if len(Xtr) < 120:
            preds.append(np.nan)
            continue
        model = LinearRegression()
        model.fit(Xtr, ytr)
        preds.append(float(model.predict(X.loc[[d]])[0]))

    model_full = LinearRegression().fit(X, y)
    last = X.iloc[-1].copy()
    fc = []
    for _ in range(horizon):
        p = float(model_full.predict(last.values.reshape(1, -1))[0])
        fc.append(p)
        for k in range(max_lag, 1, -1):
            last[f"lag{k}"] = last[f"lag{k-1}"]
        last["lag1"] = p
    return {"name": "LinReg", "test_dates": dates_test, "test_pred": np.array(preds), "fc5": np.array(fc)}


# ---- Random Forest + RandomizedSearchCV (tuning une fois) ----
def _rf_best_params(X: pd.DataFrame, y: pd.Series) -> Dict:
    tscv = TimeSeriesSplit(n_splits=3)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [None, 4, 6, 8, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }
    rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=tscv,
                            scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1, verbose=0)
    rs.fit(X, y)
    return rs.best_params_


def rf_walk_and_forecast(feat: pd.DataFrame, test_days: int = 30, horizon: int = 5, max_lag: int = 10) -> Dict:
    X, y = _build_ml_matrix(feat, max_lag=max_lag)
    idx = X.index
    dates_test = pd.DatetimeIndex(idx[-test_days:]) if len(idx) >= test_days else idx
    preds = []

    if len(idx) <= test_days + 150:
        best = {"n_estimators": 400, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}
    else:
        best = _rf_best_params(X.loc[idx < dates_test[0]], y.loc[idx < dates_test[0]])

    for d in dates_test:
        mask = idx < d
        Xtr, ytr = X.loc[mask], y.loc[mask]
        if len(Xtr) < 150:
            preds.append(np.nan)
            continue
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **best)
        model.fit(Xtr, ytr)
        preds.append(float(model.predict(X.loc[[d]])[0]))

    model_full = RandomForestRegressor(random_state=42, n_jobs=-1, **best)
    model_full.fit(X, y)
    last = X.iloc[-1].copy()
    fc = []
    for _ in range(horizon):
        p = float(model_full.predict(last.values.reshape(1, -1))[0])
        fc.append(p)
        for k in range(max_lag, 1, -1):
            last[f"lag{k}"] = last[f"lag{k-1}"]
        last["lag1"] = p
    return {"name": "RF(RS)", "test_dates": dates_test, "test_pred": np.array(preds), "fc5": np.array(fc)}


# ---- XGBoost + RandomizedSearchCV (si dispo) ----
def _xgb_best_params(X: pd.DataFrame, y: pd.Series) -> Dict:
    tscv = TimeSeriesSplit(n_splits=3)
    model = XGBRegressor(random_state=42, n_estimators=400, n_jobs=-1)
    param_dist = {
        "n_estimators": [300, 400, 600],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }
    rs = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=tscv,
                            scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1, verbose=0)
    rs.fit(X, y)
    return rs.best_params_


def xgb_walk_and_forecast(feat: pd.DataFrame, test_days: int = 30, horizon: int = 5, max_lag: int = 10) -> Dict:
    if not XGB_AVAILABLE:
        return {"name": "XGBoost (non dispo)", "test_dates": pd.DatetimeIndex([]), "test_pred": np.array([]), "fc5": np.array([])}
    X, y = _build_ml_matrix(feat, max_lag=max_lag)
    idx = X.index
    dates_test = pd.DatetimeIndex(idx[-test_days:]) if len(idx) >= test_days else idx
    preds = []

    if len(idx) <= test_days + 150:
        best = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0}
    else:
        best = _xgb_best_params(X.loc[idx < dates_test[0]], y.loc[idx < dates_test[0]])

    for d in dates_test:
        mask = idx < d
        Xtr, ytr = X.loc[mask], y.loc[mask]
        if len(Xtr) < 150:
            preds.append(np.nan)
            continue
        model = XGBRegressor(random_state=42, n_jobs=-1, **best)
        model.fit(Xtr, ytr)
        preds.append(float(model.predict(X.loc[[d]])[0]))

    model_full = XGBRegressor(random_state=42, n_jobs=-1, **best)
    model_full.fit(X, y)
    last = X.iloc[-1].copy()
    fc = []
    for _ in range(horizon):
        p = float(model_full.predict(last.values.reshape(1, -1))[0])
        fc.append(p)
        for k in range(max_lag, 1, -1):
            last[f"lag{k}"] = last[f"lag{k-1}"]
        last["lag1"] = p
    return {"name": "XGB(RS)", "test_dates": dates_test, "test_pred": np.array(preds), "fc5": np.array(fc)}


# =========================
# Réseaux de neurones TensorFlow — LSTM
# =========================
def lstm_walk_and_forecast(feat: pd.DataFrame, test_days: int = 30, horizon: int = 5, window: int = 20,
                           epochs: int = 30, batch_size: int = 64, units: int = 32) -> Dict:
    """LSTM univarié sur la série des rendements log (fenêtre glissante)."""
    if not TF_AVAILABLE:
        return {"name": "LSTM (non dispo)", "test_dates": pd.DatetimeIndex([]), "test_pred": np.array([]), "fc5": np.array([])}

    series = feat["log_ret"].dropna().astype(np.float32)
    idx = series.index

    def make_xy(arr, win):
        X, Y = [], []
        for i in range(len(arr) - win):
            X.append(arr[i:i+win])
            Y.append(arr[i+win])
        return np.array(X, dtype=np.float32)[..., None], np.array(Y, dtype=np.float32)

    arr = series.values
    X_all, y_all = make_xy(arr, window)
    idx_y = idx[window:]
    dates_test = pd.DatetimeIndex(idx_y[-test_days:]) if len(idx_y) >= test_days else idx_y

    preds = []
    for d in dates_test:
        # FIX: position robuste de d dans idx_y
        cut = idx_y.get_loc(d)  # integer location exact
        Xtr, ytr = X_all[:cut], y_all[:cut]
        if len(Xtr) < 300:
            preds.append(np.nan)
            continue
        model = Sequential([LSTM(units, input_shape=(window,1)), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(Xtr, ytr, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
        preds.append(float(model.predict(X_all[cut-1:cut], verbose=0)[0][0]))

    model_full = Sequential([LSTM(units, input_shape=(window,1)), Dense(1)])
    model_full.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model_full.fit(X_all, y_all, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

    last_win = arr[-window:].reshape(1, window, 1)
    fc = []
    for _ in range(horizon):
        p = float(model_full.predict(last_win, verbose=0)[0][0])
        fc.append(p)
        last_win = np.concatenate([last_win[:,1:,:], np.array(p, dtype=np.float32).reshape(1,1,1)], axis=1)

    return {"name": f"LSTM(TF,w={window})", "test_dates": dates_test, "test_pred": np.array(preds), "fc5": np.array(fc)}


# =========================
# Classification (proba de hausse)
# =========================
def logistic_updown_walk_and_forecast(feat: pd.DataFrame, test_days: int = 30, horizon: int = 5, max_lag: int = 10) -> Dict:
    X, y_reg = _build_ml_matrix(feat, max_lag=max_lag)
    y_cls = (y_reg > 0).astype(int)
    idx = X.index
    dates_test = pd.DatetimeIndex(idx[-test_days:]) if len(idx) >= test_days else idx

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    proba_test = []
    for d in dates_test:
        mask = idx < d
        Xtr, ytr = X.loc[mask], y_cls.loc[mask]
        if len(Xtr) < 200:
            proba_test.append(np.nan)
            continue
        pipe.fit(Xtr, ytr)
        p = float(pipe.predict_proba(X.loc[[d]])[0, 1])
        proba_test.append(p)

    # forecast 5j (récursif) via une LinReg interne pour faire avancer les lags
    lin = LinearRegression().fit(X, y_reg)
    last = X.iloc[-1].copy()
    proba_fc5 = []
    for _ in range(horizon):
        pipe.fit(X, y_cls)
        p = float(pipe.predict_proba(last.values.reshape(1, -1))[0, 1])
        proba_fc5.append(p)
        r_hat = float(lin.predict(last.values.reshape(1, -1))[0])
        for k in range(max_lag, 1, -1):
            last[f"lag{k}"] = last[f"lag{k-1}"]
        last["lag1"] = r_hat

    return {"name": "Logit(up)", "test_dates": dates_test, "proba_up_test": np.array(proba_test), "proba_up_fc5": np.array(proba_fc5)}


# =========================
# K-MEANS & FUZZY C-MEANS — Régimes + Prévisions
# =========================
def _safe_kmeans(n_clusters: int, random_state: int = 42):
    try:
        return KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    except TypeError:
        return KMeans(n_clusters=n_clusters, random_state=random_state)


def _kmeans_fit(train_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """Fit KMeans sur (vol20, rv20, mom5) avec StandardScaler."""
    cols = ["vol20", "rv20", "mom5"]
    tr = train_df[cols].dropna()
    if len(tr) < max(30, n_clusters * 5):
        return None, None
    scaler = StandardScaler().fit(tr.values)
    km = _safe_kmeans(n_clusters, random_state).fit(scaler.transform(tr.values))
    return scaler, km


def _kmeans_predict_labels(scaler: StandardScaler, km: KMeans, df_feat: pd.DataFrame) -> np.ndarray:
    """ORDRE ARGUMENTS: (scaler, km, df_feat)."""
    cols = ["vol20", "rv20", "mom5"]
    X = df_feat[cols].dropna()
    if X.empty:
        return np.array([])
    return km.predict(scaler.transform(X.values))


def _kmeans_mu_next_return(train_df: pd.DataFrame, scaler, km) -> Dict[int, float]:
    """Mu_k = E[log_ret_{t+1} | regime_t = k] estimé sur le train."""
    cols = ["vol20", "rv20", "mom5"]
    trX = train_df[cols].dropna()
    idx = trX.index.intersection(train_df.index)
    trX = trX.reindex(idx)
    y = train_df["log_ret"].reindex(idx)

    labels = km.predict(scaler.transform(trX.values))
    y_next = y.shift(-1)
    valid = y_next.dropna().index
    lab_s = pd.Series(labels, index=idx).reindex(valid)

    mu = {}
    for k in range(km.n_clusters):
        m = y_next.loc[valid][lab_s == k]
        mu[k] = float(m.mean()) if len(m) > 0 else 0.0
    return mu


def kmeans_walk_and_forecast(feat: pd.DataFrame, test_days: int = 30, horizon: int = 5, n_clusters: int = 3) -> Dict:
    """Walk-forward + forecast via régimes KMeans (persistance)."""
    idx_all = feat.index
    dates_test = pd.DatetimeIndex(idx_all[-test_days:]) if len(idx_all) >= test_days else idx_all
    preds = []

    for d in dates_test:
        train_df = feat.loc[feat.index < d]
        if len(train_df) < 60:
            preds.append(np.nan); continue
        scaler, km = _kmeans_fit(train_df, n_clusters=n_clusters)
        if scaler is None:
            preds.append(np.nan); continue
        mu = _kmeans_mu_next_return(train_df, scaler, km)
        x_df = feat.loc[[d], ["vol20", "rv20", "mom5"]].dropna()
        if x_df.empty:
            preds.append(np.nan); continue
        k_d = int(_kmeans_predict_labels(scaler, km, x_df)[0])
        preds.append(float(mu.get(k_d, 0.0)))

    window = feat.dropna().tail(min(len(feat), 250))
    scaler, km = _kmeans_fit(window, n_clusters=n_clusters)
    if scaler is None:
        fc5 = np.array([0.0] * horizon)
    else:
        mu = _kmeans_mu_next_return(window, scaler, km)
        x_last = window[["vol20", "rv20", "mom5"]].iloc[[-1]].dropna()
        if x_last.empty:
            fc5 = np.array([0.0] * horizon)
        else:
            k_last = int(_kmeans_predict_labels(scaler, km, x_last)[0])
            fc5 = np.array([float(mu.get(k_last, 0.0))] * horizon)

    return {"name": f"KMeans(k={n_clusters})", "test_dates": dates_test, "test_pred": np.array(preds), "fc5": fc5}


# ---------- Fuzzy C-Means (léger) ----------
def _fcm_train(X: np.ndarray, n_clusters: int = 3, m: float = 2.0, max_iter: int = 150, tol: float = 1e-5, random_state: int = 42):
    """Retourne (centroïdes V[k], U membership N×K)."""
    rng = np.random.default_rng(random_state)
    N, d = X.shape; K = n_clusters
    try:
        km = _safe_kmeans(K, random_state).fit(X)
        U = np.zeros((N, K), dtype=float)
        U[np.arange(N), km.labels_] = 1.0
        U = np.clip(U + 1e-6, 1e-6, 1.0); U = U / U.sum(axis=1, keepdims=True)
    except Exception:
        U = rng.random((N, K)); U = U / U.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        Um = U ** m
        denom = Um.sum(axis=0, keepdims=True)
        denom = np.where(denom == 0.0, 1e-12, denom)
        V = (Um.T @ X) / denom.T  # K×d

        dist = np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2) + 1e-12  # N×K
        inv = 1.0 / dist
        inv_pow = inv ** (2.0 / (m - 1.0))
        U_new = inv_pow / inv_pow.sum(axis=1, keepdims=True)

        if np.max(np.abs(U_new - U)) < tol:
            U = U_new; break
        U = U_new

    return V, U


def _fcm_membership_for_x(V: np.ndarray, x: np.ndarray, m: float = 2.0) -> np.ndarray:
    dist = np.linalg.norm(V - x[None, :], axis=1) + 1e-12
    inv = 1.0 / dist
    inv_pow = inv ** (2.0 / (m - 1.0))
    return inv_pow / np.sum(inv_pow)


def fcm_walk_and_forecast(feat: pd.DataFrame, test_days: int = 30, horizon: int = 5,
                          n_clusters: int = 3, m: float = 1.8, max_iter: int = 100, tol: float = 1e-5,
                          random_state: int = 42) -> Dict:
    """Walk-forward + forecast via FCM (membership flou + persistance)."""
    cols = ["vol20", "rv20", "mom5"]
    idx_all = feat.index
    dates_test = pd.DatetimeIndex(idx_all[-test_days:]) if len(idx_all) >= test_days else idx_all
    preds = []

    for d in dates_test:
        train_df = feat.loc[feat.index < d]
        trX = train_df[cols].dropna()
        if len(trX) < max(40, n_clusters * 6):
            preds.append(np.nan); continue

        X = trX.values.astype(float)
        V, U = _fcm_train(X, n_clusters=n_clusters, m=m, max_iter=max_iter, tol=tol, random_state=random_state)

        idx = trX.index
        y = train_df["log_ret"].reindex(idx)
        y_next = y.shift(-1)
        valid = y_next.dropna().index
        Uv = U[:len(valid), :]
        yv = y_next.loc[valid].values.reshape(-1, 1)
        Um = Uv ** m
        denom = Um.sum(axis=0, keepdims=True)
        denom = np.where(denom == 0.0, 1e-12, denom)
        mu_k = (Um.T @ yv).flatten() / denom.flatten()

        x_df = feat.loc[[d], cols].dropna()
        if x_df.empty:
            preds.append(np.nan); continue
        x = x_df.values.astype(float)[0]
        u_x = _fcm_membership_for_x(V, x, m=m)
        num = np.sum((u_x ** m) * mu_k); den = np.sum(u_x ** m) + 1e-12
        preds.append(float(num / den))

    # Forecast horizon
    window = feat.dropna().tail(min(len(feat), 250))
    trX = window[cols].dropna()
    if len(trX) < max(40, n_clusters * 6):
        fc5 = np.array([0.0] * horizon)
    else:
        X = trX.values.astype(float)
        V, U = _fcm_train(X, n_clusters=n_clusters, m=m, max_iter=max_iter, tol=tol, random_state=random_state)

        idx = trX.index
        y = window["log_ret"].reindex(idx)
        y_next = y.shift(-1)
        valid = y_next.dropna().index
        Uv = U[:len(valid), :]
        yv = y_next.loc[valid].values.reshape(-1, 1)
        Um = Uv ** m
        denom = Um.sum(axis=0, keepdims=True)
        denom = np.where(denom == 0.0, 1e-12, denom)
        mu_k = (Um.T @ yv).flatten() / denom.flatten()

        x_last = window[cols].iloc[[-1]].dropna()
        if x_last.empty:
            fc5 = np.array([0.0] * horizon)
        else:
            x = x_last.values.astype(float)[0]
            u_x = _fcm_membership_for_x(V, x, m=m)
            r_hat = float(np.sum((u_x ** m) * mu_k) / (np.sum(u_x ** m) + 1e-12))
            fc5 = np.array([r_hat] * horizon)

    return {"name": f"FCM(k={n_clusters}, m={m})", "test_dates": dates_test, "test_pred": np.array(preds), "fc5": fc5}


# =========================
# Régimes (K-Means) — étiquetage pour visualisation/pondération
# =========================
def label_regimes_for_dates(feat: pd.DataFrame, dates: pd.DatetimeIndex, n_clusters: int = 3) -> pd.Series:
    labels = []
    for d in dates:
        train_df = feat.loc[feat.index < d]
        scaler, km = _kmeans_fit(train_df, n_clusters=n_clusters)
        if scaler is None:
            labels.append(np.nan); continue
        x_df = feat.loc[[d], ["vol20", "rv20", "mom5"]].dropna()
        if x_df.empty:
            labels.append(np.nan); continue
        lab = _kmeans_predict_labels(scaler, km, x_df)  # ordre FIXE: (scaler, km, x_df)
        labels.append(float(lab[0]) if len(lab) else np.nan)
    return pd.Series(labels, index=dates, name="regime")


def current_regime_label(feat: pd.DataFrame, n_clusters: int = 3) -> float:
    if len(feat) < 60:
        return np.nan
    window = feat.dropna().tail(min(250, len(feat)-1))
    train_df = window.iloc[:-1]
    x_df = window.iloc[[-1]][["vol20", "rv20", "mom5"]].dropna()
    scaler, km = _kmeans_fit(train_df, n_clusters=n_clusters)
    if scaler is None or x_df.empty:
        return np.nan
    lab = _kmeans_predict_labels(scaler, km, x_df)
    return float(lab[0]) if len(lab) else np.nan


# =========================
# Évaluation & utilitaires
# =========================
def compute_backtest_metrics(actual: pd.Series, pred: pd.Series) -> Dict[str, float]:
    a = pd.Series(actual).astype(float)
    p = pd.Series(pred).astype(float)
    m = pd.concat([a, p], axis=1).dropna()
    if m.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "SIGN_ACC": np.nan}
    err = m.iloc[:, 0] - m.iloc[:, 1]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sign_acc = float((np.sign(m.iloc[:, 0]) == np.sign(m.iloc[:, 1])).mean() * 100.0)
    return {"MAE": mae, "RMSE": rmse, "SIGN_ACC": sign_acc}


def compute_mae_by_regime(actual: pd.Series, preds: pd.Series, regimes: pd.Series) -> Dict[int, float]:
    out = {}
    for r in sorted(regimes.dropna().unique()):
        mask = (regimes == r)
        a = actual.loc[mask]
        p = preds.loc[mask]
        m = pd.concat([a, p], axis=1).dropna()
        out[int(r)] = (float(np.mean(np.abs(m.iloc[:, 0] - m.iloc[:, 1]))) if not m.empty else np.nan)
    return out


def ensemble_fc5(weights: Dict[str, float], fc_map: Dict[str, np.ndarray]) -> np.ndarray:
    s = sum(max(w, 0) for w in weights.values())
    valid = [v for v in fc_map.values() if v is not None and len(v) > 0]
    if s <= 0 or not valid:
        return np.mean(valid, axis=0) if valid else np.zeros(5)
    wnorm = {k: max(v, 0) / s for k, v in weights.items()}
    first = valid[0]
    combo = np.zeros_like(first)
    for k, v in fc_map.items():
        if v is not None and len(v) == len(first):
            combo += wnorm.get(k, 0.0) * v
    return combo


def project_price_from_returns(last_price: float, returns: np.ndarray) -> np.ndarray:
    price_path = [last_price]
    for r in returns:
        price_path.append(price_path[-1] * np.exp(r))
    return np.array(price_path[1:])


# =========================
# GARCH (VOLATILITÉ)
# =========================
def fit_garch(log_ret: pd.Series, horizon: int = 5) -> Dict:
    """GARCH(1,1) pour la vol/jour en % (sert pour l'incertitude)."""
    r = (log_ret.dropna() - log_ret.dropna().mean()) * 100.0
    am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    fc = res.forecast(horizon=horizon)
    vol = np.sqrt(fc.variance.iloc[-1].values)  # %/jour
    return {"name": "GARCH(1,1)", "vol_forecast_pct": vol, "params": res.params.to_dict()}
