"""Supply-chain monitor: free, key-less feeds mapped onto each commodity's physical flow.

Weather triggers are scored by the risk engine (Open-Meteo). These feeds add direct evidence of
disruption at ports, rivers, sea lanes and fields. They drive the flow diagrams, the advisor's
reasons and the end-of-day note; they do not change position sizing.
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote, urlencode

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "millenium-poc/0.1 (commodity supply-chain research)"}
PORTWATCH = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
UNKNOWN, NORMAL, WATCH, BREACH, CRITICAL = -1, 0, 1, 2, 3
LEVEL_NAME = {UNKNOWN: "Not connected", NORMAL: "Normal", WATCH: "Watch", BREACH: "Breach", CRITICAL: "Critical"}

PORTS = {"port1160": "Santos", "port56": "Antofagasta", "port728": "Mejillones", "port264": "Corpus Christi",
         "port1214": "South Louisiana", "port812": "New Orleans"}
CHOKEPOINTS = {"chokepoint2": "Panama Canal"}
# USGS gauge -> name, then low-water (watch, breach, extreme) and high-water (watch, breach, extreme) stage in ft.
# Approximate barge-operations levels for a PoC, not official navigation thresholds.
RIVER_GAUGES = {
    "07010000": ("Mississippi at St. Louis", (0, -3, -8), (25, 30, 38)),
    "07374000": ("Mississippi at Baton Rouge", (8, 4, 1), (30, 35, 42)),
}
BARGE_RATES_DATASET = "7spn-fbua"     # USDA AgTransport: Mississippi downbound grain barge $/ton
DROUGHT_STATES = {"KS": 20, "TX": 48}  # US Drought Monitor area-of-interest = state FIPS
DROUGHT_SEASON = (5, 6, 7, 8, 9)       # sorghum growing season; off-season drought is capped at Watch
US_FACILITIES = ("donaldsonville", "haynesville", "florida", "kansas", "texas_hp", "corpus")
GDACS_RADIUS_KM = {"TC": 500, "FL": 300, "EQ": 300, "DR": 600, "VO": 200, "WF": 150}
NEWS_QUERIES = {
    "NH3": '(ammonia OR "nitrogen fertilizer") (plant OR outage OR shutdown OR hurricane)',
    "LI": 'lithium (Atacama OR Antofagasta OR SQM OR Chile)',
    "FCOJ": '("orange juice" OR citrus) (Florida OR Brazil OR Santos OR freeze)',
    "SORG": 'sorghum (Kansas OR Texas OR export OR China OR drought)',
}


@dataclass
class Signal:
    key: str
    label: str
    level: int
    detail: str
    source: str
    as_of: str | None = None
    url: str | None = None


@dataclass
class FeedStatus:
    name: str
    provides: str
    ok: bool
    message: str
    as_of: str | None = None
    n_signals: int = 0


@dataclass
class SupplyChainSnapshot:
    signals: dict[str, Signal] = field(default_factory=dict)
    feeds: list[FeedStatus] = field(default_factory=list)
    news: dict[str, list[dict]] = field(default_factory=dict)


# --- pure scoring helpers ------------------------------------------------------------------------
def ratio_level(ratio: float, watch: float, breach: float, higher_is_worse: bool = False) -> int:
    if ratio is None or not math.isfinite(ratio):
        return UNKNOWN
    if higher_is_worse:
        return BREACH if ratio >= breach else WATCH if ratio >= watch else NORMAL
    return BREACH if ratio <= breach else WATCH if ratio <= watch else NORMAL


def activity_drop(daily: pd.Series, recent_days: int = 14, baseline_days: int = 90) -> tuple[float, float, float]:
    """Recent mean / baseline mean of a daily count series (port calls, canal transits)."""
    daily = daily.sort_index()
    recent, baseline = daily.tail(recent_days), daily.iloc[-(recent_days + baseline_days):-recent_days]
    base = baseline.mean()
    return (recent.mean() / base if base and base > 0 else float("nan")), float(recent.mean()), float(base)


def stage_level(value: float, low: tuple, high: tuple) -> int:
    if value is None or not math.isfinite(value):
        return UNKNOWN
    if value <= low[2] or value >= high[2]:
        return CRITICAL
    if value <= low[1] or value >= high[1]:
        return BREACH
    if value <= low[0] or value >= high[0]:
        return WATCH
    return NORMAL


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


# --- feeds ---------------------------------------------------------------------------------------
class SupplyChainMonitor:
    def __init__(self, session: requests.Session | None = None, timeout: float = 30):
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout

    def _get(self, url: str, retries: int = 1, **params):
        for attempt in range(retries + 1):
            resp = self.session.get(url, params=params or None, timeout=self.timeout)
            if resp.status_code < 500 or attempt == retries:
                resp.raise_for_status()
                return resp.json()
            time.sleep(1.5 * (attempt + 1))

    def _arcgis(self, layer: str, where: str | list[str], fields: str, order: str, n: int, retries: int = 5) -> list[dict]:
        """PortWatch's ArcGIS layers intermittently answer 'Invalid URL'; %20-encode, retry, alternate filters."""
        wheres = [where] if isinstance(where, str) else where
        error = None
        for attempt in range(retries):
            query = urlencode({"where": wheres[attempt % len(wheres)], "outFields": fields, "orderByFields": order,
                               "resultRecordCount": n, "returnGeometry": "false", "f": "json"}, quote_via=quote)
            data = self.session.get(f"{PORTWATCH}/{layer}/FeatureServer/0/query?{query}", timeout=self.timeout).json()
            if "error" not in data:
                return [f["attributes"] for f in data.get("features", [])]
            error = data["error"].get("message")
            time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"PortWatch {layer}: {error}")

    def fetch(self, prices: pd.DataFrame | None = None) -> SupplyChainSnapshot:
        feeds = {
            "IMF PortWatch · port calls": ("Daily AIS-derived port calls (6 export ports)", self.portwatch_ports),
            "IMF PortWatch · chokepoints": ("Daily Panama Canal transits", self.portwatch_chokepoints),
            "IMF PortWatch · disruptions": ("Disaster events flagged against our ports", self.portwatch_disruptions),
            "USGS Water Services": ("Mississippi River stage (barge low / high water)", self.usgs_river),
            "USDA AgTransport": ("Mississippi grain barge rates ($/ton)", self.barge_rates),
            "US Drought Monitor": ("Kansas / Texas drought area (D2+)", self.drought_monitor),
            "NOAA NHC": ("Active Atlantic tropical cyclones vs US facilities", self.nhc_storms),
            "NWS alerts": ("Active watches / warnings at US facilities", self.nws_alerts),
            "GDACS": ("Global disaster alerts near every facility", self.gdacs_events),
        }
        snap = SupplyChainSnapshot()
        with ThreadPoolExecutor(max_workers=len(feeds) + 1) as pool:
            futures = {name: pool.submit(fn) for name, (_, fn) in feeds.items()}
            news_future = pool.submit(self.gdelt_news)
            for name, (provides, _) in feeds.items():
                try:
                    signals, as_of = futures[name].result()
                    snap.signals.update({s.key: s for s in signals})
                    snap.feeds.append(FeedStatus(name, provides, True, "ok", as_of, len(signals)))
                except Exception as exc:  # each feed fails alone; the flow marks its nodes "unavailable"
                    log.warning("supply-chain feed %s failed: %s", name, exc)
                    snap.feeds.append(FeedStatus(name, provides, False, f"{type(exc).__name__}: {str(exc)[:120]}"))
            try:
                snap.news, msg = news_future.result()
                snap.feeds.append(FeedStatus("GDELT DOC 2.0", "News headlines per commodity", not msg, msg or "ok",
                                             datetime.now(timezone.utc).date().isoformat(),
                                             sum(len(v) for v in snap.news.values())))
            except Exception as exc:
                snap.feeds.append(FeedStatus("GDELT DOC 2.0", "News headlines per commodity", False, str(exc)[:120]))
        if prices is not None and "NG=F" in prices:
            snap.signals["market:NG=F"] = self.gas_price_signal(prices["NG=F"])
        snap.signals["dcsa:containers"] = Signal(
            "dcsa:containers", "Container schedules (DCSA)", UNKNOWN,
            "Needs a carrier API key (e.g. Maersk / Hapag-Lloyd DCSA Commercial Schedules or Track & Trace)", "DCSA")
        return snap

    def portwatch_ports(self):
        with ThreadPoolExecutor(max_workers=len(PORTS)) as pool:
            fetched = dict(zip(PORTS, pool.map(
                lambda pid: self._arcgis("Daily_Ports_Data", f"portid='{pid}'", "date,portcalls", "date DESC", 130), PORTS)))
        out, as_of = [], None
        for pid, name in PORTS.items():
            rows = pd.DataFrame(fetched[pid])
            if rows.empty:
                continue
            series = rows.assign(date=pd.to_datetime(rows["date"])).set_index("date")["portcalls"].astype(float)
            series = series.resample("D").sum()
            as_of = max(as_of or "", series.index.max().date().isoformat())
            ratio, recent, base = activity_drop(series)
            out.append(Signal(f"port:{pid}", f"{name} port calls", ratio_level(ratio, 0.7, 0.5),
                              f"{recent:.1f}/day last 14d vs {base:.1f}/day prior 90d ({ratio:.0%})",
                              "IMF PortWatch", series.index.max().date().isoformat(),
                              f"https://portwatch.imf.org/pages/{pid}"))
        return out, as_of

    def portwatch_chokepoints(self):
        out, as_of = [], None
        for cid, name in CHOKEPOINTS.items():
            wheres = [f"portid='{cid}' AND year>={date.today().year - 1}", f"portid='{cid}'"]
            rows = pd.DataFrame(self._arcgis("Daily_Chokepoints_Data", wheres, "*", "date DESC", 130, retries=8))
            if rows.empty:
                raise ValueError(f"no rows for {cid}")
            series = rows.assign(date=pd.to_datetime(rows["date"])).set_index("date")["n_total"].astype(float).sort_index()
            ratio, recent, base = activity_drop(series)
            as_of = series.index.max().date().isoformat()
            out.append(Signal(f"chokepoint:{cid}", f"{name} transits", ratio_level(ratio, 0.8, 0.6),
                              f"{recent:.1f}/day last 14d vs {base:.1f}/day prior 90d ({ratio:.0%})",
                              "IMF PortWatch", as_of, f"https://portwatch.imf.org/pages/{cid}"))
        return out, as_of

    def portwatch_disruptions(self):
        events = self._arcgis("portwatch_disruptions_database", "1=1",
                              "eventtype,eventname,alertlevel,fromdate,todate,affectedports", "fromdate DESC", 300)
        cutoff = datetime.now(timezone.utc) - timedelta(days=14)
        hits: dict[str, list] = {}
        for a in events:
            end = datetime.fromtimestamp((a.get("todate") or 0) / 1000, tz=timezone.utc)
            if end < cutoff:
                continue
            for pid in str(a.get("affectedports") or "").replace(" ", "").split(","):
                if pid in PORTS:
                    hits.setdefault(pid, []).append(a)
        out = []
        for pid, name in PORTS.items():
            events = hits.get(pid, [])
            level = max((BREACH if e.get("alertlevel", "").upper() == "RED" else WATCH for e in events), default=NORMAL)
            detail = "; ".join(f"{e['eventname']} ({e['alertlevel']})" for e in events[:2]) or "No active events flagged"
            out.append(Signal(f"portevent:{pid}", f"{name} disruption events", level, detail, "IMF PortWatch / GDACS"))
        return out, datetime.now(timezone.utc).date().isoformat()

    def usgs_river(self):
        data = self._get("https://waterservices.usgs.gov/nwis/iv/", sites=",".join(RIVER_GAUGES),
                         parameterCd="00065", period="P30D", format="json")
        out, as_of = [], None
        for ts in data["value"]["timeSeries"]:
            site = ts["sourceInfo"]["siteCode"][0]["value"]
            values = [(v["dateTime"], float(v["value"])) for v in ts["values"][0]["value"] if v["value"] not in ("", "-999999")]
            if site not in RIVER_GAUGES or not values:
                continue
            name, low, high = RIVER_GAUGES[site]
            stamp, stage = values[-1]
            lo30, hi30 = min(v for _, v in values), max(v for _, v in values)
            as_of = stamp[:16]
            out.append(Signal(f"river:{site}", f"{name} stage", stage_level(stage, low, high),
                              f"{stage:.1f} ft now (30d range {lo30:.1f}–{hi30:.1f}); low-water breach ≤ {low[1]} ft, "
                              f"high-water ≥ {high[1]} ft", "USGS", as_of,
                              f"https://waterdata.usgs.gov/monitoring-location/{site}/"))
        return out, as_of

    def barge_rates(self):
        rows = self._get(f"https://agtransport.usda.gov/resource/{BARGE_RATES_DATASET}.json",
                         **{"$where": f"date >= '{(date.today() - timedelta(days=400)).isoformat()}'",
                            "$limit": 5000})
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("no barge rate rows")
        df["date"] = pd.to_datetime(df["date"])
        df["price_per_ton"] = pd.to_numeric(df["price_per_ton"], errors="coerce")
        # Cape Girardeau–Grafton and Memphis segments bracket St. Louis on the middle Mississippi.
        stl = df[df["river_system_location"].str.contains("Grafton|Memphis", case=False, na=False)]
        series = (stl if not stl.empty else df).groupby("date")["price_per_ton"].mean().sort_index()
        latest, median = series.iloc[-1], series.tail(52).median()
        ratio = latest / median if median else float("nan")
        as_of = series.index[-1].date().isoformat()
        return [Signal("barge:mississippi", "Mississippi grain barge rate", ratio_level(ratio, 1.25, 1.6, higher_is_worse=True),
                       f"${latest:.2f}/ton vs 52-week median ${median:.2f} ({ratio:.0%})"
                       + (" · Grafton–Memphis segments" if not stl.empty else " · all segments"),
                       "USDA AgTransport", as_of, f"https://agtransport.usda.gov/d/{BARGE_RATES_DATASET}")], as_of

    def drought_monitor(self):
        end = date.today()
        out, as_of = [], None
        for state, fips in DROUGHT_STATES.items():
            resp = self.session.get(
                "https://usdmdataservices.unl.edu/api/StateStatistics/GetDroughtSeverityStatisticsByAreaPercent",
                params={"aoi": fips, "startdate": (end - timedelta(days=60)).strftime("%-m/%-d/%Y"),
                        "enddate": end.strftime("%-m/%-d/%Y"), "statisticsType": 1},
                headers={"Accept": "application/json"}, timeout=self.timeout)
            resp.raise_for_status()
            rows = sorted(resp.json(), key=lambda r: r["mapDate"])
            if not rows:
                raise ValueError(f"no drought rows for {state}")
            latest, prior = rows[-1], rows[max(0, len(rows) - 5)]
            d2 = float(latest["d2"])
            level = ratio_level(d2, 20, 40, higher_is_worse=True)
            in_season = end.month in DROUGHT_SEASON
            if not in_season:
                level = min(level, WATCH)
            as_of = latest["mapDate"][:10]
            out.append(Signal(f"drought:{state}", f"{state} drought area (D2+)", level,
                              f"{d2:.0f}% of state in severe+ drought (4 weeks ago {float(prior['d2']):.0f}%)"
                              + ("" if in_season else "; off-season for sorghum"),
                              "US Drought Monitor", as_of, "https://droughtmonitor.unl.edu/"))
        return out, as_of

    def nhc_storms(self):
        storms = [s for s in self._get("https://www.nhc.noaa.gov/CurrentStorms.json").get("activeStorms", [])
                  if str(s.get("id", "")).startswith("al")]
        out = []
        for key in ("donaldsonville", "florida", "corpus"):
            fac = config.FACILITIES[key]
            best = None
            for s in storms:
                d = haversine_km(fac.lat, fac.lon, float(s["latitudeNumeric"]), float(s["longitudeNumeric"]))
                if best is None or d < best[0]:
                    best = (d, s)
            if best is None:
                out.append(Signal(f"nhc:{key}", "Atlantic tropical cyclones", NORMAL, "No active Atlantic storms", "NOAA NHC"))
                continue
            d, s = best
            hurricane = float(s.get("intensity") or 0) >= 64
            level = (CRITICAL if hurricane and d < 200 else BREACH if d < 350 else WATCH if d < 900 else NORMAL)
            out.append(Signal(f"nhc:{key}", "Atlantic tropical cyclones", level,
                              f"{s.get('classification')} {s.get('name')} {s.get('intensity')} kt, {d:,.0f} km away",
                              "NOAA NHC", s.get("lastUpdate"), "https://www.nhc.noaa.gov/"))
        return out, datetime.now(timezone.utc).date().isoformat()

    def nws_alerts(self):
        out = []
        for key in US_FACILITIES:
            fac = config.FACILITIES[key]
            feats = self._get("https://api.weather.gov/alerts/active", point=f"{fac.lat},{fac.lon}").get("features", [])
            props = [f["properties"] for f in feats]
            def level_of(p):
                event = p.get("event", "")
                return BREACH if "Warning" in event else WATCH if ("Watch" in event or "Advisory" in event) else NORMAL
            level = max((level_of(p) for p in props), default=NORMAL)
            detail = "; ".join(p.get("event", "") for p in props[:3]) or "No active alerts"
            out.append(Signal(f"nws:{key}", "NWS watches / warnings", level, detail, "NWS",
                              datetime.now(timezone.utc).isoformat(timespec="minutes"),
                              f"https://forecast.weather.gov/MapClick.php?lat={fac.lat}&lon={fac.lon}"))
        return out, datetime.now(timezone.utc).date().isoformat()

    def gdacs_events(self):
        data = self._get("https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH", retries=2,
                         eventlist="TC;FL;EQ;DR;VO;WF", alertlevel="Orange;Red",
                         fromdate=(date.today() - timedelta(days=21)).isoformat())
        events = []
        for f in data.get("features", []):
            p, (lon, lat) = f["properties"], f["geometry"]["coordinates"][:2]
            if str(p.get("todate", ""))[:10] >= (date.today() - timedelta(days=3)).isoformat():
                events.append((p, lat, lon))
        out = []
        for key, fac in config.FACILITIES.items():
            near = [(p, haversine_km(fac.lat, fac.lon, lat, lon)) for p, lat, lon in events]
            near = [(p, d) for p, d in near if d <= GDACS_RADIUS_KM.get(p.get("eventtype"), 300)]
            level = max((BREACH if p.get("alertlevel") == "Red" else WATCH for p, _ in near), default=NORMAL)
            detail = "; ".join(f"{p.get('name') or p.get('eventname')} ({p.get('alertlevel')}, {d:,.0f} km)"
                               for p, d in near[:2]) or "No Orange/Red alerts nearby"
            out.append(Signal(f"gdacs:{key}", "GDACS disaster alerts", level, detail, "GDACS",
                              datetime.now(timezone.utc).date().isoformat(), "https://www.gdacs.org/"))
        return out, datetime.now(timezone.utc).date().isoformat()

    def gdelt_news(self, per_query: int = 5) -> tuple[dict[str, list[dict]], str | None]:
        """GDELT allows one request per 5 s; on a 429 the feed reports rate-limited and returns nothing."""
        news, problem = {}, None
        for i, (asset, query) in enumerate(NEWS_QUERIES.items()):
            if i:
                time.sleep(5.5)
            resp = self.session.get("https://api.gdeltproject.org/api/v2/doc/doc", timeout=self.timeout, params={
                "query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
                "maxrecords": per_query, "timespan": "3d", "sort": "datedesc"})
            if resp.status_code == 429:
                return news, "rate-limited by GDELT (1 request / 5 s per IP); retry later"
            resp.raise_for_status()
            try:
                articles = resp.json().get("articles", [])
            except ValueError:
                articles, problem = [], "GDELT returned a non-JSON response"
            news[asset] = [{"title": a.get("title"), "url": a.get("url"), "source": a.get("domain"),
                            "seen": a.get("seendate")} for a in articles]
        return news, problem

    @staticmethod
    def gas_price_signal(ng: pd.Series) -> Signal:
        ng = ng.dropna()
        change = ng.iloc[-1] / ng.iloc[-6] - 1
        return Signal("market:NG=F", "Henry Hub gas (NG=F) 5-day move",
                      ratio_level(change, 0.12, 0.25, higher_is_worse=True),
                      f"${ng.iloc[-1]:.2f}/MMBtu, {change:+.0%} over 5 sessions", "Yahoo Finance",
                      ng.index[-1].date().isoformat())


# --- physical flows ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    upstream: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()   # weather triggers (names in config.ASSETS)
    signals: tuple[str, ...] = ()    # supply-chain signal keys


FLOWS: dict[str, tuple[Stage, ...]] = {
    "NH3": (
        Stage("gas", "Natural gas feedstock · Haynesville", triggers=("Wellhead freeze-off",),
              signals=("market:NG=F", "nws:haynesville")),
        Stage("plant", "NH3 synthesis · Donaldsonville, LA", ("gas",),
              triggers=("Gulf hurricane wind", "Plant freeze / power loss"),
              signals=("nhc:donaldsonville", "nws:donaldsonville", "gdacs:donaldsonville")),
        Stage("river", "Barge distribution · Mississippi", ("plant",),
              signals=("river:07374000", "river:07010000", "barge:mississippi")),
        Stage("port", "Gulf export · South Louisiana / New Orleans", ("plant",),
              signals=("port:port1214", "port:port812", "portevent:port1214", "portevent:port812")),
        Stage("sea", "Sea lane · Panama Canal", ("port",), signals=("chokepoint:chokepoint2",)),
        Stage("market", "Market · CF (NH3 proxy)", ("river", "sea")),
    ),
    "LI": (
        Stage("brine", "Brine evaporation · Salar de Atacama", triggers=("Salar flash flood / rainfall",),
              signals=("gdacs:atacama",)),
        Stage("plant", "Carbonate / hydroxide plants · Antofagasta", ("brine",), signals=("gdacs:antofagasta",)),
        Stage("port", "Export ports · Antofagasta / Mejillones", ("plant",), triggers=("Port wind closure",),
              signals=("port:port56", "port:port728", "portevent:port56", "portevent:port728")),
        Stage("ship", "Container shipping · to Asia", ("port",), signals=("dcsa:containers",)),
        Stage("market", "Market · LIT ETF", ("ship",)),
    ),
    "FCOJ": (
        Stage("fl", "Groves · Central Florida", triggers=("Florida hard freeze", "Florida hurricane wind"),
              signals=("nhc:florida", "nws:florida", "gdacs:florida")),
        Stage("sp", "Groves · São Paulo belt",
              triggers=("São Paulo drought (water-balance anomaly)", "São Paulo cyclone wind"), signals=("gdacs:saopaulo",)),
        Stage("plant", "Juice processing & cold storage", ("fl", "sp")),
        Stage("port", "Export port · Santos", ("plant",), triggers=("Santos heavy rain (loading halts)",),
              signals=("port:port1160", "portevent:port1160", "gdacs:santos")),
        Stage("market", "Market · ICE FCOJ (OJ=F)", ("port",)),
    ),
    "SORG": (
        Stage("ks", "Fields · Western Kansas", triggers=("Kansas flowering heat", "Kansas drought (water-balance anomaly)"),
              signals=("drought:KS", "nws:kansas")),
        Stage("tx", "Fields · Texas High Plains", triggers=("Texas High Plains heat",), signals=("drought:TX", "nws:texas_hp")),
        Stage("inland", "Inland rail / barge · to Gulf", ("ks", "tx"), signals=("barge:mississippi", "river:07010000")),
        Stage("port", "Export elevators · Corpus Christi", ("inland",),
              triggers=("Texas Gulf hurricane (export elevators)",),
              signals=("port:port264", "portevent:port264", "nhc:corpus", "nws:corpus")),
        Stage("sea", "Sea lane · Panama Canal → China", ("port",), signals=("chokepoint:chokepoint2",)),
        Stage("market", "Market · CBOT corn (ZC=F)", ("sea",)),
    ),
}


@dataclass
class RiskItem:
    name: str
    kind: str          # "weather" or "feed"
    level: int
    reading: str
    source: str
    why: str = ""
    url: str | None = None


@dataclass
class StageRisk:
    stage: Stage
    level: int
    items: list[RiskItem]


def weather_level(severity: float, p_breach: float | None) -> int:
    score = max(severity, p_breach or 0.0)
    if score >= 0.75:
        return CRITICAL
    if score >= 0.5:
        return BREACH
    return WATCH if severity > 0 or (p_breach or 0) >= 0.10 else NORMAL


def evaluate_flow(asset_key: str, report, snapshot: SupplyChainSnapshot | None) -> list[StageRisk]:
    readings = {r.name: r for r in report.signals[asset_key].readings}
    probs = {t.trigger: t for t in report.trigger_probabilities if t.asset == asset_key}
    impacts = {t.name: t.impact for t in config.ASSETS[asset_key].triggers}
    out = []
    for stage in FLOWS[asset_key]:
        items = []
        for name in stage.triggers:
            r, tp = readings.get(name), probs.get(name)
            p = tp.p_breach if tp else None
            reading = (f"P(breach 8d) {p:.0%} · " if p is not None else "") + (
                f"peak {r.peak_value:.1f} {r.unit} vs {r.threshold:g}" if r else "")
            if r is not None and not r.in_season:
                reading += " · off-season"
            items.append(RiskItem(name, "weather", weather_level(r.severity if r else 0.0, p), reading,
                                  "Open-Meteo ensemble", impacts.get(name, "")))
        for key in stage.signals:
            s = snapshot.signals.get(key) if snapshot else None
            items.append(RiskItem(s.label, "feed", s.level, s.detail, s.source, url=s.url) if s else
                         RiskItem(key.split(":")[0], "feed", UNKNOWN, "feed unavailable", "—"))
        known = [i.level for i in items if i.level != UNKNOWN]
        level = max(known) if known else (UNKNOWN if items else NORMAL)
        out.append(StageRisk(stage, level, items))
    return out
