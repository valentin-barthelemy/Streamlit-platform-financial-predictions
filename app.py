import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import core

st.set_page_config(page_title="Plateforme boursière — Régimes + Deep Learning", layout="wide")
st.title("📈 Plateforme d'analyse — Backtest 30j, Régimes & Prévision 5j")

# ========= Aide erreurs (UX) =========
class _ShowErr:
    @staticmethod
    def show_user_error(e: Exception):
        if hasattr(core, "UXError") and isinstance(e, core.UXError):
            titles = {"data":"Erreur de données","features":"Erreur de préparation des features",
                      "model":"Erreur d'entraînement / environnement","prediction":"Erreur de prédiction"}
            title = titles.get(getattr(e, "kind", ""), "Erreur")
            with st.expander(f"❌ {title} — détails (clique pour ouvrir)", expanded=True):
                st.error(str(e))
                det = getattr(e, "details", "")
                if det: st.code(det, language="text")
        else:
            with st.expander("❌ Erreur inattendue — détails", expanded=True):
                st.exception(e)

# ========= CACHES =========
@st.cache_data(show_spinner=False, ttl=3600)
def _cached_load_prices(ticker, period, interval):
    return core.load_prices(ticker, period=period, interval=interval)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_features(prices_df):
    return core.compute_features(prices_df)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_company_name(ticker):
    return core.get_company_name(ticker)

# --- Tickers proposés
TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","NFLX","IBM","ORCL","INTC","AMD","AVGO","ADBE","CRM","CSCO",
    "JPM","BAC","WFC","GS","V","MA","KO","PEP","DIS","NKE","PG","HD","MCD","COST","WMT",
    "AIR.PA","OR.PA","SAN.PA","BNP.PA","MC.PA","AI.PA","SU.PA","RMS.PA","DG.PA","GLE.PA","VIE.PA","ENGI.PA",
    "ASML.AS","AD.AS","SAP.DE","BAYN.DE","SIE.DE","IBE.MC","ITX.MC","ENEL.MI","STM.MI",
    "^GSPC","^NDX","^DJI","SPY","QQQ","IWM","XOM","CVX","BP","TTE.PA"
]

def sanitize_ticker(raw: str) -> str:
    if not raw: return ""
    t = raw.strip().upper()
    for sep in [",",";"," "]:
        if sep in t: t = t.split(sep)[0]; break
    return t

# ========= Helpers UI / Stats =========
def _make_regime_bands(idx: pd.DatetimeIndex, regimes: pd.Series):
    shapes = []
    if regimes.isna().all() or len(idx) == 0: return shapes
    pal = ["rgba(66,135,245,0.10)","rgba(245,66,93,0.10)","rgba(66,245,161,0.10)","rgba(245,196,66,0.10)","rgba(155,66,245,0.10)"]
    regs = list(regimes.values); current = None; start = None
    for i, (t, r) in enumerate(zip(idx, regs)):
        if np.isnan(r): continue
        if current is None:
            current, start = int(r), t
        elif int(r) != current:
            shapes.append(dict(type="rect", xref="x", yref="paper", x0=start, x1=idx[i-1], y0=0, y1=1,
                               fillcolor=pal[current % len(pal)], line=dict(width=0)))
            current, start = int(r), t
    if current is not None:
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=start, x1=idx[-1], y0=0, y1=1,
                           fillcolor=pal[current % len(pal)], line=dict(width=0)))
    return shapes

def _prob_up_from_residuals(bt_df: pd.DataFrame, model_name: str, fc_returns: np.ndarray):
    """Proba naïve de hausse (somme des retours vs bruit résiduel du modèle)."""
    if model_name not in bt_df.columns: return float("nan"), float("nan"), np.array([])
    m = bt_df[["Actual", model_name]].dropna()
    if len(m) < 15: return float("nan"), float("nan"), np.array([])
    resid = (m["Actual"] - m[model_name]).values
    sigma1 = float(np.std(resid, ddof=1))
    mu_H = float(np.sum(fc_returns)); sigma_H = sigma1 * math.sqrt(len(fc_returns))
    from math import erf, sqrt
    Phi = 0.5 * (1.0 + erf((mu_H / (sigma_H if sigma_H>0 else 1e-12)) / sqrt(2.0)))
    return float(np.clip(Phi, 0.0, 1.0)), sigma1, resid

def _gated_meta_ensemble(bt_df: pd.DataFrame, regime: float, eligible: dict):
    """Poids non-négatifs appris sur le backtest du régime courant (projection NNLS simple)."""
    names = [k for k in eligible.keys() if k in bt_df.columns]
    mask = (bt_df["Régime"] == int(regime)) if regime==regime else (bt_df["Actual"]==bt_df["Actual"])
    df = bt_df.loc[mask, ["Actual"] + names].dropna()
    if df.shape[0] < 20 or len(names) == 0:
        fc = np.mean([v for v in eligible.values()], axis=0) if eligible else np.zeros(5)
        return fc, {}
    y = df["Actual"].values; X = df[names].values
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    w = np.clip(w, 0, None); s = w.sum()
    if s <= 0: w = np.ones_like(w); s = w.sum()
    w = w / s
    H = len(next(iter(eligible.values()))); combo = np.zeros(H)
    for wi, name in zip(w, names):
        v = eligible[name]
        if v is not None and len(v) == H: combo += wi * v
    return combo, {n: float(wi) for n, wi in zip(names, w)}

# ========= Calibration robuste (outliers) =========
def _clip_probs(p, eps: float = 1e-3):
    p = np.asarray(p, dtype=float); return np.clip(p, eps, 1.0 - eps)

def _bin_and_smooth(probs: np.ndarray, y: np.ndarray, n_bins: int = 10, alpha: float = 1.0):
    df = pd.DataFrame({"p": probs, "y": y}).dropna().sort_values("p")
    bins = np.array_split(df, n_bins); x_b, y_b, w_b = [], [], []
    for b in bins:
        if len(b) == 0: continue
        p_med = float(b["p"].median()); n = int(len(b)); k = float(b["y"].sum())
        y_hat = (k + alpha) / (n + 2 * alpha)  # Laplace
        x_b.append(p_med); y_b.append(y_hat); w_b.append(n)
    return np.array(x_b), np.array(y_b), np.array(w_b, dtype=float)

def _brier(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float); p_hat = np.asarray(p_hat, dtype=float)
    return float(np.mean((p_hat - y_true) ** 2))

def _robust_calibrator(proba_bt: pd.Series, y_bt: pd.Series):
    align = proba_bt.index.intersection(y_bt.index)
    p = _clip_probs(proba_bt.reindex(align).values, eps=1e-3)
    y = y_bt.reindex(align).values.astype(int)
    n = len(p)
    if n < 50:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        def predict(p_new): return iso.predict(_clip_probs(np.array(p_new), 1e-3))
        return predict, {"method":"iso(fallback)","iso_points":(p, y, np.ones_like(p))}
    cut = max(30, int(0.8 * n))
    p_tr, y_tr = p[:cut], y[:cut]; p_va, y_va = p[cut:], y[cut:]
    x_b, y_b, w_b = _bin_and_smooth(p_tr, y_tr, n_bins=10, alpha=1.0)
    iso = IsotonicRegression(out_of_bounds="clip").fit(x_b, y_b, sample_weight=w_b)
    def iso_predict(p_new): return iso.predict(_clip_probs(np.array(p_new), 1e-3))
    def _logit(x): x = _clip_probs(np.array(x), 1e-3); return np.log(x / (1.0 - x))
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(p_tr).reshape(-1,1), y_tr)
    def platt_predict(p_new): return lr.predict_proba(_logit(p_new).reshape(-1,1))[:,1]
    if len(p_va) >= 10:
        b_iso = _brier(y_va, iso_predict(p_va)); b_pl = _brier(y_va, platt_predict(p_va))
        method = "platt" if b_pl + 1e-4 < b_iso else ("iso(binned)" if b_iso + 1e-4 < b_pl else "avg")
    else:
        method = "iso(binned)"
    def predict(p_new):
        if method == "platt": return platt_predict(p_new)
        if method == "iso(binned)": return iso_predict(p_new)
        p_iso = iso_predict(p_new); p_pl = platt_predict(p_new); return 0.5 * p_iso + 0.5 * p_pl
    return predict, {"method":method, "iso_points":(x_b, y_b, w_b)}

# ========= Bandes 80% =========
def _bootstrap_band(last_price: float, fc: np.ndarray, resid: np.ndarray, n_paths: int = 1000, alpha: float = 0.20, seed: int = 42):
    rng = np.random.default_rng(seed)
    H = len(fc)
    if resid is None or len(resid) < 5:
        return np.full(H, np.nan), np.full(H, np.nan)
    noise = rng.choice(resid, size=(n_paths, H), replace=True)
    cum = np.cumsum(fc.reshape(1, -1) + noise, axis=1)
    paths = last_price * np.exp(cum)
    lower = np.quantile(paths, alpha/2.0, axis=0)
    upper = np.quantile(paths, 1.0 - alpha/2.0, axis=0)
    return lower, upper

# ========= Barre latérale =========
with st.sidebar:
    st.header("⚙️ Paramètres")
    choice = st.selectbox("Sélectionne un ticker", TICKERS + ["Autre (manuel)"], index=(TICKERS.index("RMS.PA") if "RMS.PA" in TICKERS else 0))
    ticker = sanitize_ticker(st.text_input("Ticker manuel", value="RMS.PA")) if choice=="Autre (manuel)" else choice
    period = st.selectbox("Historique", ["1y","2y","3y","5y"], index=2)
    interval = st.selectbox("Intervalle", ["1d","1wk"], index=0)
    test_days = st.slider("Backtest (jours ouvrés)", 20, 60, 30)
    horizon = st.slider("Prévision (jours ouvrés)", 1, 10, 5)
    n_clusters = st.slider("Nombre de régimes (K-Means/FCM)", 2, 5, 3)
    band_type = st.selectbox("Type d'intervalle 80%", ["Bootstrap (résidus)", "Résidus + GARCH"], index=0)
    n_boot = st.slider("Bootstrap (chemins simulés)", 200, 3000, 1000, step=200)
    run = st.button("Lancer l'analyse", type="primary")

progress = st.progress(0)
phase = st.empty()

# ========= RUN =========
if run:
    try:
        if not ticker:
            st.error("Merci de sélectionner/saisir un ticker valide."); st.stop()

        # 1) Données / features
        phase.info("Téléchargement des données…")
        prices = _cached_load_prices(ticker, period, interval)
        feat = _cached_features(prices)
        last_price = float(feat["close"].iloc[-1])
        full_name = _cached_company_name(ticker)
        progress.progress(12)
        st.subheader(f"🔎 Analyse : **{full_name}**")

        # 2) Modèles — y compris KMeans + FCM
        errors = {}

        try: sar = core.sarima_walk_and_forecast(feat["log_ret"], test_days=test_days, horizon=horizon, order=(1,1,1))
        except Exception as e: errors["SARIMA"]=e; sar={"name":"SARIMA","test_dates":pd.DatetimeIndex([]),"test_pred":np.array([]),"fc5":np.array([])}
        progress.progress(24)

        try: lin = core.linreg_walk_and_forecast(feat, test_days=test_days, horizon=horizon, max_lag=10)
        except Exception as e: errors["LinReg"]=e; lin={"name":"LinReg","test_dates":pd.DatetimeIndex([]),"test_pred":np.array([]),"fc5":np.array([])}
        progress.progress(36)

        try: rf = core.rf_walk_and_forecast(feat, test_days=test_days, horizon=horizon, max_lag=10)
        except Exception as e: errors["RF(RS)"]=e; rf={"name":"RF(RS)","test_dates":pd.DatetimeIndex([]),"test_pred":np.array([]),"fc5":np.array([])}
        progress.progress(48)

        try: xgb = core.xgb_walk_and_forecast(feat, test_days=test_days, horizon=horizon, max_lag=10)
        except Exception as e: errors["XGB(RS)"]=e; xgb={"name":"XGB(RS)","test_dates":pd.DatetimeIndex([]),"test_pred":np.array([]),"fc5":np.array([])}
        progress.progress(60)

        try: lstm = core.lstm_walk_and_forecast(feat, test_days=test_days, horizon=horizon, window=20, epochs=40, batch_size=64, units=32)
        except Exception as e: errors["LSTM(TF)"]=e; lstm={"name":"LSTM(TF)","test_dates":pd.DatetimeIndex([]),"test_pred":np.array([]),"fc5":np.array([])}
        progress.progress(72)

        try: km = core.kmeans_walk_and_forecast(feat, test_days=test_days, horizon=horizon, n_clusters=int(n_clusters))
        except Exception as e: errors["KMeans"]=e; km={"name":f"KMeans(k={n_clusters})","test_dates":pd.DatetimeIndex([]),"test_pred":np.array([]),"fc5":np.array([])}
        progress.progress(82)

        try: fcm = core.fcm_walk_and_forecast(feat, test_days=test_days, horizon=horizon, n_clusters=int(n_clusters), m=1.8)
        except Exception as e: errors["FCM"]=e; fcm={"name":f"FCM(k={n_clusters}, m=1.8)","test_dates":pd.DatetimeIndex([]),"test_pred":np.array([]),"fc5":np.array([])}
        progress.progress(88)

        try: gar = core.fit_garch(feat["log_ret"], horizon=horizon)
        except Exception as e: errors["GARCH"]=e; gar={"name":"GARCH(1,1)","vol_forecast_pct":np.array([np.nan]*horizon),"params":{}}
        progress.progress(94)

        # 3) Backtest anchor (alignement)
        def ser_from(dct):
            if len(dct.get("test_dates", [])) == 0: return pd.Series(dtype=float)
            return pd.Series(dct["test_pred"], index=pd.DatetimeIndex(dct["test_dates"]))

        pred_series = {
            "SARIMA": ser_from(sar), "LinReg": ser_from(lin), "RF(RS)": ser_from(rf),
            "XGB(RS)": ser_from(xgb), "LSTM(TF)": ser_from(lstm),
            km["name"]: ser_from(km), fcm["name"]: ser_from(fcm)
        }
        counts = {k:int(s.notna().sum()) for k,s in pred_series.items() if len(s)>0}
        if not counts:
            raise core.UXError("prediction", "Aucune prédiction de backtest exploitable (trop peu d'historique ?).")
        anchor_name = max(counts, key=counts.get)
        back_idx = pred_series[anchor_name].index

        bt_df = pd.DataFrame(index=back_idx)
        bt_df.index.name = "Date"
        bt_df["Actual"] = feat["log_ret"].reindex(back_idx).astype(float)
        for name, s in pred_series.items():
            if len(s)>0: bt_df[name] = s.reindex(back_idx)

        # Régimes (visu + pondération)
        regimes = core.label_regimes_for_dates(feat, back_idx, n_clusters=int(n_clusters))
        bt_df["Régime"] = regimes
        cur_regime = core.current_regime_label(feat, n_clusters=int(n_clusters))

        # Métriques
        metrics = {}
        for col in [c for c in bt_df.columns if c not in ["Actual","Régime"]]:
            metrics[col] = core.compute_backtest_metrics(bt_df["Actual"], bt_df[col])

        # 4) Prévisions (récursives)
        fut_idx = pd.date_range(feat.index[-1], periods=horizon+1, freq="B")[1:]
        fc_map = {
            "SARIMA": sar["fc5"],
            "LinReg": lin["fc5"],
            "RF(RS)": rf["fc5"],
            "XGB(RS)": xgb["fc5"] if len(xgb["fc5"])>0 else None,
            "LSTM(TF)": lstm["fc5"] if len(lstm["fc5"])>0 else None,
            km["name"]: km["fc5"] if len(km["fc5"])>0 else None,
            fcm["name"]: fcm["fc5"] if len(fcm["fc5"])>0 else None,
        }
        eligible = {k:v for k,v in fc_map.items() if v is not None and len(v)>0}
        valid_metrics = {k:metrics[k] for k in eligible.keys() if k in metrics}

        # Meilleur modèle (SIGN_ACC → MAE)
        best_name = max(
            (valid_metrics.keys() or ["SARIMA"]),
            key=lambda k: (valid_metrics.get(k,{"SIGN_ACC":-1})["SIGN_ACC"], -valid_metrics.get(k,{"MAE":np.inf})["MAE"])
        )

        # Ensembles
        def _ensemble(weights, fc_map):
            s = sum(max(w,0) for w in weights.values())
            if s <= 0: return np.mean(list(fc_map.values()), axis=0)
            wn = {k:max(v,0)/s for k,v in weights.items()}
            H = len(next(iter(fc_map.values()))); combo = np.zeros(H)
            for k,v in fc_map.items():
                if v is not None and len(v)==H: combo += wn.get(k,0.0)*v
            return combo

        wg = {}
        for name, met in valid_metrics.items():
            mae = met.get("MAE", np.nan)
            if np.isfinite(mae) and mae > 0: wg[name] = 1.0/mae
        fc_ens = _ensemble(wg, eligible)

        wr = {}
        for name, arr in eligible.items():
            if name not in bt_df.columns: continue
            mae_by_r = core.compute_mae_by_regime(bt_df["Actual"], bt_df[name], bt_df["Régime"])
            m_r = mae_by_r.get(int(cur_regime)) if cur_regime==cur_regime else None
            m_g = metrics.get(name,{}).get("MAE", np.nan)
            denom = m_r if (m_r is not None and np.isfinite(m_r) and m_r>0) else m_g
            if np.isfinite(denom) and denom>0: wr[name] = 1.0/denom
        fc_ens_reg = _ensemble(wr, eligible)
        fc_gate, w_gate = _gated_meta_ensemble(bt_df, cur_regime, eligible)

        # Prix projetés
        price_proj = {name: core.project_price_from_returns(last_price, arr) for name, arr in eligible.items()}
        price_proj["Ensemble"] = core.project_price_from_returns(last_price, fc_ens)
        price_proj["Ensemble (régime)"] = core.project_price_from_returns(last_price, fc_ens_reg)
        price_proj["Ensemble (gated)"] = core.project_price_from_returns(last_price, fc_gate)

        # Proba de hausse & résidus du meilleur
        best_fc = eligible[best_name]
        p_up, sigma_res, resid = _prob_up_from_residuals(bt_df, best_name, best_fc)

        # Calibration robuste
        p_cal_curve = None
        try:
            logit = core.logistic_updown_walk_and_forecast(feat, test_days=test_days, horizon=horizon, max_lag=10)
            proba_bt = pd.Series(logit["proba_up_test"], index=logit["test_dates"]).dropna()
            y_bt = (feat["log_ret"].reindex(proba_bt.index) > 0).astype(int).dropna()
            align = proba_bt.index.intersection(y_bt.index)
            proba_bt = proba_bt.reindex(align); y_bt = y_bt.reindex(align)
            if len(proba_bt) >= 30 and y_bt.notna().sum() == len(proba_bt):
                predict_cal, cal_info = _robust_calibrator(proba_bt, y_bt)
                if np.isfinite(p_up): p_up = float(predict_cal([p_up])[0])
                p_cal_curve = ("robust", cal_info, proba_bt, y_bt, predict_cal)
        except Exception:
            pass

        # Bandes 80%
        if band_type.startswith("Bootstrap"):
            lower_b, upper_b = _bootstrap_band(last_price, best_fc, resid, n_paths=n_boot, alpha=0.20, seed=42)
        else:
            garch_arr = gar.get("vol_forecast_pct", None)
            z = 1.2815515655446004
            mu_cum = np.cumsum(best_fc); H = len(best_fc); days = np.arange(1, H+1)
            sigma1 = float(np.std(resid, ddof=1)) if len(resid)>5 else 0.0
            if garch_arr is not None and len(garch_arr) >= H and np.isfinite(garch_arr).all():
                garch_std = (np.array(garch_arr[:H], dtype=float) / 100.0)
                sigma_path = np.sqrt(np.cumsum((sigma1**2) + (garch_std**2)))
            else:
                sigma_path = sigma1 * np.sqrt(days)
            upper_b = last_price * np.exp(mu_cum + z * sigma_path)
            lower_b = last_price * np.exp(mu_cum - z * sigma_path)

        # HEADLINE
        best_path = price_proj[best_name]
        price_tH = best_path[-1]; chgH = (price_tH/last_price - 1.0) * 100.0
        prob_txt = f"{(p_up*100):.1f}%" if np.isfinite(p_up) else "—"
        st.markdown(
            f"""
            <div style="text-align:center; margin: 10px 0 24px 0;">
              <div style="font-size: 28px; line-height:1; font-weight:700;">Δ {chgH:+.2f}%</div>
              <div style="font-size: 44px; font-weight:800; margin-top:8px;">{full_name} — Prix prévu dans {horizon} j : {price_tH:,.2f}</div>
              <div style="font-size: 20px; color:#666; margin-top:8px;">Prob(hausse): {prob_txt} — Modèle: {best_name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # TABS
        tab1, tab2, tab3 = st.tabs(["Résumé", "Régimes & perfs", "Calibration proba"])

        with tab1:
            st.subheader("📊 Backtest 30 jours — métriques & 🔔 Volatilité (GARCH)")
            c1, c2 = st.columns([1,1])
            with c1:
                mtab = pd.DataFrame(
                    [[k, v["MAE"], v["RMSE"], v["SIGN_ACC"]] for k, v in metrics.items()],
                    columns=["Modèle","MAE","RMSE","Précision signe (%)"]
                ).sort_values("MAE")
                st.dataframe(mtab.style.format({"MAE":"{:.6f}","RMSE":"{:.6f}","Précision signe (%)":"{:.1f}"}), height=360)
            with c2:
                gar_tbl = pd.DataFrame({"t+": np.arange(1, horizon+1), "vol_%/jour": gar["vol_forecast_pct"]})
                st.dataframe(gar_tbl.round(3), height=360)

            st.subheader("📉 Backtest — actual vs prédictions (30 jours) + régimes")
            fig_bt = go.Figure()
            shapes = _make_regime_bands(bt_df.index, bt_df["Régime"])
            fig_bt.update_layout(shapes=shapes)
            fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Actual"], name="Actual", mode="lines+markers"))
            for col in [c for c in bt_df.columns if c not in ["Actual","Régime"]]:
                if bt_df[col].notna().any():
                    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df[col], name=col, mode="lines+markers"))
            fig_bt.update_layout(height=600, margin=dict(l=8,r=8,t=30,b=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_bt, use_container_width=True)
            st.caption("Bandes colorées = régimes K-Means (vol/momentum).")

            st.subheader(f"📈 Cours + trajectoires ({horizon} jours) — {band_type}")
            hist_tail = feat["close"].tail(10)
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="Historique (Close)", mode="lines"))
            fig_fc.add_trace(go.Scatter(x=fut_idx, y=upper_b, name="Upper", mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_fc.add_trace(go.Scatter(x=fut_idx, y=lower_b, name="Intervalle 80%", mode="lines", fill="tonexty", line=dict(width=0)))
            fig_fc.add_trace(go.Scatter(x=fut_idx, y=best_path, name=f"{best_name}", mode="lines+markers"))
            fig_fc.add_trace(go.Scatter(x=fut_idx, y=price_proj["Ensemble (gated)"], name="Ensemble (gated)", mode="lines+markers"))
            fig_fc.update_layout(height=480, margin=dict(l=8,r=8,t=30,b=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_fc, use_container_width=True)

            st.subheader("💡 Prévisions détaillées (t+1 → t+5)")
            c3, c4, c5 = st.columns([1,1,1])
            with c3:
                fc_table = pd.DataFrame(index=fut_idx)
                for name, arr in eligible.items(): fc_table[name] = arr
                fc_table["Ensemble"] = fc_ens; fc_table["Ensemble (régime)"] = fc_ens_reg; fc_table["Ensemble (gated)"] = fc_gate
                st.dataframe(fc_table.round(6), height=260)
            with c4:
                price_table = pd.DataFrame(index=fut_idx)
                for name, arr in price_proj.items(): price_table[name] = arr
                st.dataframe(price_table.round(2), height=260)
            with c5:
                st.markdown("**Régime courant**")
                st.write(int(cur_regime) if cur_regime==cur_regime else "—")
                if w_gate:
                    st.markdown("**Poids (meta-ensemble, régime courant)**")
                    st.json({k: round(v,3) for k,v in w_gate.items()})

        with tab2:
            st.subheader("🧭 Performance par régime (MAE)")
            byreg = {}
            for name in [c for c in bt_df.columns if c not in ["Actual","Régime"]]:
                byreg[name] = core.compute_mae_by_regime(bt_df["Actual"], bt_df[name], bt_df["Régime"])
            regs = sorted({r for d in byreg.values() for r in d.keys()})
            mat = pd.DataFrame(index=regs, columns=byreg.keys(), dtype=float)
            for mname, dct in byreg.items():
                for r, v in dct.items(): mat.loc[r, mname] = v
            st.dataframe(mat.round(6))
            st.caption("Choisis un modèle en fonction du régime où il excelle (MAE faible).")

        with tab3:
            st.subheader("📏 Calibration de la probabilité (robuste aux outliers)")
            if 'p_cal_curve' in locals() and p_cal_curve is not None:
                _, cal_info, proba_bt, y_bt, predict_cal = p_cal_curve
                x_b, y_b, w_b = cal_info.get("iso_points", (None, None, None))
                grid = np.linspace(0.01, 0.99, 100); curve = predict_cal(grid)
                fig = go.Figure()
                if x_b is not None and y_b is not None:
                    fig.add_trace(go.Scatter(x=x_b, y=y_b, mode="markers", marker=dict(size=8), name="Bins (Laplace-smoothed)"))
                fig.add_trace(go.Scatter(x=grid, y=curve, mode="lines", name=f"Calibrateur ({cal_info['method']})"))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Idéal (y=x)"))
                fig.update_layout(height=380, margin=dict(l=8,r=8,t=20,b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Clipping [0.001, 0.999], binning égal-fréquence + Laplace, sélection (Isotonic/Platt) par Brier score.")
            else:
                st.info("Pas assez d’observations pour calibrer (ou Logit indisponible).")

        if errors:
            with st.expander("⚠️ Avertissements / erreurs par modèle", expanded=True):
                for name, err in errors.items():
                    st.markdown(f"**{name}**"); _ShowErr.show_user_error(err)

        progress.progress(100); phase.success("Terminé.")

    except Exception as e:
        phase.empty(); _ShowErr.show_user_error(e)
else:
    st.info("Choisis un ticker puis clique **Lancer l'analyse**.")
