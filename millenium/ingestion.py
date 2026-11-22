"""Data ingestion: Open-Meteo weather (no API key) and Yahoo Finance via yfinance."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from . import config
from .config import Facility

log = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENSEMBLE_VARS = ("temperature_2m_min", "precipitation_sum", "wind_speed_10m_max")
DAILY_VARS = (
    "temperature_2m_min", "temperature_2m_max", "precipitation_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max", "et0_fao_evapotranspiration",
)
CURRENT_VARS = ("temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m")


@dataclass
class WeatherSnapshot:
    daily: dict[str, pd.DataFrame]     # facility key -> daily frame indexed by date
    current: dict[str, dict]           # facility key -> current conditions
    today: dict[str, date]             # facility-local "today"
    climatology: dict[str, pd.Series] = field(default_factory=dict)   # day-of-year -> normal 30d water balance
    # facility -> metric -> frame (forecast date x ensemble member), pooled across models
    ensemble: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    ensemble_models: tuple[str, ...] = ()


class OpenMeteoClient:
    def __init__(self, session: requests.Session | None = None, timeout: float = 20):
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch(self, facilities: list[Facility], climatology: dict[str, pd.Series] | None = None) -> WeatherSnapshot:
        climatology = climatology or {}
        # Open-Meteo accepts comma-separated coordinates, so one request covers every facility.
        params = {
            "latitude": ",".join(str(f.lat) for f in facilities),
            "longitude": ",".join(str(f.lon) for f in facilities),
            "daily": ",".join(DAILY_VARS),
            "current": ",".join(CURRENT_VARS),
            "past_days": config.PAST_DAYS,
            "forecast_days": config.FORECAST_DAYS,
            "timezone": "auto",
            "wind_speed_unit": "kmh",
        }
        resp = self.session.get(OPEN_METEO_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            payload = [payload]

        daily, current, today = {}, {}, {}
        for fac, loc in zip(facilities, payload):
            frame = pd.DataFrame(loc["daily"])
            frame["time"] = pd.to_datetime(frame["time"]).dt.date
            frame = frame.set_index("time")
            daily[fac.key] = add_derived_metrics(frame, climatology.get(fac.key))
            current[fac.key] = loc.get("current", {})
            # past_days rows precede the facility-local current day
            today[fac.key] = frame.index[config.PAST_DAYS]
        return WeatherSnapshot(daily=daily, current=current, today=today, climatology=climatology)

    def fetch_climatology(self, facilities: list[Facility], years: tuple[int, int] = config.CLIMATOLOGY_YEARS
                          ) -> dict[str, pd.Series]:
        """Seasonal normal of the 30-day water balance by day of year, smoothed over ±7 days."""
        history = self.fetch_history(facilities, date(years[0], 1, 1), date(years[1], 12, 31))
        out = {}
        for key, frame in history.items():
            balance = add_derived_metrics(frame)["water_balance_30d"]
            by_doy = balance.groupby(pd.to_datetime(balance.index).dayofyear).mean().reindex(range(1, 367)).ffill()
            wrapped = pd.concat([by_doy.tail(7), by_doy, by_doy.head(7)])
            out[key] = wrapped.rolling(15, center=True).mean().iloc[7:-7]
        return out

    def fetch_history(self, facilities: list[Facility], start: date, end: date) -> dict[str, pd.DataFrame]:
        """Observed (ERA5 reanalysis) daily weather; the archive lags real time by ~5 days."""
        params = {
            "latitude": ",".join(str(f.lat) for f in facilities),
            "longitude": ",".join(str(f.lon) for f in facilities),
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": ",".join(DAILY_VARS), "timezone": "auto", "wind_speed_unit": "kmh",
        }
        resp = self.session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=max(self.timeout, 60))
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            payload = [payload]
        out = {}
        for fac, loc in zip(facilities, payload):
            frame = pd.DataFrame(loc["daily"])
            frame["time"] = pd.to_datetime(frame["time"]).dt.date
            out[fac.key] = frame.set_index("time")
        return out


def merge_history(history: dict[str, pd.DataFrame], snapshot: WeatherSnapshot) -> dict[str, pd.DataFrame]:
    """Archive rows first, then the forecast endpoint's observed past days to close the archive lag."""
    merged = {}
    for key, hist in history.items():
        recent = snapshot.daily[key]
        recent = recent.loc[recent.index <= snapshot.today[key], list(DAILY_VARS)]
        merged[key] = add_derived_metrics(hist.combine_first(recent).sort_index(), snapshot.climatology.get(key))
    return merged


def add_derived_metrics(frame: pd.DataFrame, climatology: pd.Series | None = None) -> pd.DataFrame:
    frame = frame.copy()
    balance = frame["precipitation_sum"].fillna(0) - frame["et0_fao_evapotranspiration"].fillna(0)
    frame["water_balance_30d"] = balance.rolling(30, min_periods=20).sum()
    # Without a seasonal baseline the anomaly is unknown, which scores as no signal.
    normal = (climatology.reindex(pd.to_datetime(frame.index).dayofyear).to_numpy()
              if climatology is not None else np.nan)
    frame["water_balance_anom_30d"] = frame["water_balance_30d"] - normal
    return frame


class EnsembleClient:
    """Open-Meteo ensemble forecasts; every member is a plausible weather path."""

    def __init__(self, session: requests.Session | None = None, timeout: float = 30):
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch(self, facilities: list[Facility], models: tuple[str, ...] = config.ENSEMBLE_MODELS
              ) -> tuple[dict[str, dict[str, pd.DataFrame]], tuple[str, ...]]:
        pooled: dict[str, dict[str, list[pd.DataFrame]]] = {f.key: {m: [] for m in ENSEMBLE_VARS} for f in facilities}
        used = []
        for model in models:
            try:
                payload = self._request(facilities, model)
            except Exception as exc:  # one model down should not blank the probabilities
                log.warning("ensemble model %s unavailable: %s", model, exc)
                continue
            used.append(model)
            for fac, loc in zip(facilities, payload):
                frame = pd.DataFrame(loc["daily"])
                frame["time"] = pd.to_datetime(frame["time"]).dt.date
                frame = frame.set_index("time")
                for metric in ENSEMBLE_VARS:
                    cols = [c for c in frame.columns if c == metric or c.startswith(metric + "_member")]
                    members = frame[cols].rename(columns=lambda c: f"{model}:{c.replace(metric, '').lstrip('_') or 'control'}")
                    pooled[fac.key][metric].append(members)
        ensemble = {fac: {m: pd.concat(frames, axis=1) for m, frames in metrics.items() if frames}
                    for fac, metrics in pooled.items()}
        return ensemble, tuple(used)

    def _request(self, facilities: list[Facility], model: str) -> list[dict]:
        params = {
            "latitude": ",".join(str(f.lat) for f in facilities),
            "longitude": ",".join(str(f.lon) for f in facilities),
            "daily": ",".join(ENSEMBLE_VARS), "models": model,
            "forecast_days": config.FORECAST_DAYS, "timezone": "auto", "wind_speed_unit": "kmh",
        }
        resp = self.session.get(OPEN_METEO_ENSEMBLE_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else [payload]


@dataclass
class MarketSnapshot:
    prices: pd.DataFrame               # daily closes, columns = tickers
    implied_vol: dict[str, float | None]
    option_chains: dict[str, dict]     # option ticker -> {"expiry", "calls", "puts", "spot"}


class MarketDataClient:
    def fetch(self, tickers: list[str], option_tickers: list[str], period: str = "2y") -> MarketSnapshot:
        raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
        prices = prices.dropna(how="all")

        implied_vol, chains = {}, {}
        for t in option_tickers:
            try:
                chain = self._near_month_chain(t, spot=float(prices[t].dropna().iloc[-1]))
            except Exception as exc:  # network or empty chain: degrade to the realized-vol proxy
                log.warning("option chain unavailable for %s: %s", t, exc)
                chain = None
            if chain:
                chains[t] = chain
                implied_vol[t] = atm_implied_vol(chain)
            else:
                implied_vol[t] = None
        return MarketSnapshot(prices=prices, implied_vol=implied_vol, option_chains=chains)

    @staticmethod
    def monthly_average(ticker: str, start: str = "2000-01-01", scale: float = 1.0) -> pd.Series:
        """Monthly mean of daily closes, comparable to USDA's monthly-average cash prices."""
        raw = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False)
        close = raw["Close"]
        close = close[ticker] if isinstance(close, pd.DataFrame) else close
        return (close.dropna() * scale).resample("MS").mean()

    @staticmethod
    def _near_month_chain(ticker: str, spot: float, target_days: int = 30) -> dict | None:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return None
        today = date.today()
        dated = [(e, (date.fromisoformat(e) - today).days) for e in expiries]
        dated = [d for d in dated if d[1] >= 14] or dated
        expiry = min(dated, key=lambda d: abs(d[1] - target_days))[0]
        oc = tk.option_chain(expiry)
        return {"expiry": expiry, "calls": oc.calls, "puts": oc.puts, "spot": spot}


ERS_FEED_GRAINS_URL = "https://www.ers.usda.gov/media/5766/feed-grains-yearbook-tables-all-years.csv"
BUSHEL_LB = {"Sorghum": 56.0, "Corn": 56.0}
MONTHS = {m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), start=1)}


class FeedGrainsClient:
    """USDA ERS Feed Grains Database (free bulk CSV): monthly sorghum and corn prices in $/bu."""

    SERIES = {  # output column -> (commodity, attribute, geography)
        "sorghum_farm": ("Sorghum", "Price received by farmers", "United States"),
        "corn_farm": ("Corn", "Price received by farmers", "United States"),
        "sorghum_ks_cash": ("Sorghum, No. 2 yellow", "Cash price", "Central Kansas, KS"),
        "corn_kc_cash": ("Corn, No. 2 yellow", "Cash price", "Kansas City, MO"),
    }

    def __init__(self, url: str = ERS_FEED_GRAINS_URL, timeout: float = 120):
        self.url, self.timeout = url, timeout

    def fetch(self) -> pd.DataFrame:
        raw = pd.read_csv(self.url, encoding="latin-1", low_memory=False,
                          storage_options={"User-Agent": "Mozilla/5.0"})
        raw = raw[raw["frequency"] == "Monthly"]
        out = {}
        for col, (commodity, attribute, geography) in self.SERIES.items():
            rows = raw[(raw["commodity"] == commodity) & (raw["attribute"] == attribute) & (raw["geography"] == geography)]
            per_bu = rows["unit"].str.contains("bushel", case=False)
            rows = rows[per_bu] if per_bu.any() else rows.assign(   # $/cwt -> $/bu
                amount=pd.to_numeric(rows["amount"], errors="coerce") * BUSHEL_LB[commodity.split(",")[0]] / 100)
            rows = rows[rows["timeperiod"].isin(MONTHS)]
            idx = pd.to_datetime({"year": rows["year"].astype(int), "month": rows["timeperiod"].map(MONTHS), "day": 1})
            out[col] = pd.Series(pd.to_numeric(rows["amount"], errors="coerce").to_numpy(), index=idx).groupby(level=0).last()
        return pd.DataFrame(out).sort_index()


WORLD_BANK_COMMODITIES_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
WORLD_BANK_PINK_SHEET_FALLBACK = ("https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/"
                                  "related/CMO-Historical-Data-Monthly.xlsx")


class WorldBankPinkSheetClient:
    """World Bank Commodity Price Data ("Pink Sheet"), monthly nominal USD prices since 1960."""

    def fetch(self, columns: tuple[str, ...]) -> pd.DataFrame:
        url = WORLD_BANK_PINK_SHEET_FALLBACK
        try:  # the file URL changes with each monthly release; take the current link from the landing page
            page = requests.get(WORLD_BANK_COMMODITIES_PAGE, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text
            found = re.findall(r'https://thedocs\.worldbank\.org/[^"\']*CMO-Historical-Data-Monthly\.xlsx', page)
            url = found[0] if found else url
        except requests.RequestException:
            pass
        raw = pd.read_excel(url, sheet_name="Monthly Prices", header=4, skiprows=[5],
                            storage_options={"User-Agent": "Mozilla/5.0"})
        raw = raw.rename(columns={raw.columns[0]: "month"})
        idx = pd.to_datetime(raw["month"].astype(str).str.replace("M", "-"), format="%Y-%m", errors="coerce")
        return raw[list(columns)].apply(pd.to_numeric, errors="coerce").set_index(idx).loc[lambda d: d.index.notna()]


class BLSClient:
    """BLS public API v1 (no key; 25 requests/day, 10 years per request): CPI average retail prices."""

    URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"

    def fetch(self, series_id: str, start_year: int) -> pd.Series:
        values = {}
        for first in range(start_year, date.today().year + 1, 10):
            resp = requests.post(self.URL, json={"seriesid": [series_id], "startyear": str(first),
                                                 "endyear": str(min(first + 9, date.today().year))}, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            if body.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS: {body.get('message')}")
            for row in body["Results"]["series"][0]["data"]:
                if row["period"].startswith("M") and row["period"] != "M13" and row["value"] not in ("-", ""):
                    values[pd.Timestamp(int(row["year"]), int(row["period"][1:]), 1)] = float(row["value"])
        return pd.Series(values, name=series_id).sort_index()


def option_mid(row: pd.Series) -> float:
    bid, ask = row.get("bid", np.nan), row.get("ask", np.nan)
    if pd.notna(bid) and pd.notna(ask) and ask > 0:
        return float((bid + ask) / 2)
    return float(row.get("lastPrice", np.nan))


def atm_implied_vol(chain: dict, n_strikes: int = 3) -> float | None:
    """Median IV of the strikes nearest spot across calls and puts, ignoring Yahoo's junk prints."""
    spot = chain["spot"]
    ivs = []
    for side in ("calls", "puts"):
        df = chain[side]
        if df is None or df.empty:
            continue
        near = df.iloc[(df["strike"] - spot).abs().argsort()[:n_strikes]]
        ivs.extend(near["impliedVolatility"].tolist())
    ivs = [v for v in ivs if 0.05 < v < 3.0]
    return float(np.median(ivs)) if ivs else None


def realized_vol(prices: pd.Series, window: int) -> float:
    """Annualized close-to-close volatility over the last `window` returns."""
    rets = np.log(prices.dropna()).diff().dropna().tail(window)
    return float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 else float("nan")


def date_window(today: date, lookback_days: int | None = None, horizon_days: int | None = None) -> tuple[date, date]:
    lookback = config.LOOKBACK_OBSERVED_DAYS if lookback_days is None else lookback_days
    horizon = config.FORECAST_DAYS - 1 if horizon_days is None else horizon_days
    return today - timedelta(days=lookback), today + timedelta(days=horizon)
