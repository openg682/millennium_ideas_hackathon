"""Static universe: facilities, disruption triggers, instruments and risk limits.

Each trigger maps a daily weather metric onto a 0..1 severity ramp:
    watch     -> 0.0  (conditions start to matter)
    threshold -> 0.5  (the disruption level from the spec, e.g. wind > 65 km/h)
    extreme   -> 1.0  (near-certain physical disruption)

`bias` is the direction the trigger pushes the asset's trade instrument
(+1 LONG, -1 SHORT). These are hypotheses to be backtested, not facts.
"""
from __future__ import annotations

from dataclasses import dataclass

ABOVE, BELOW = "above", "below"


@dataclass(frozen=True)
class Facility:
    key: str
    name: str
    lat: float
    lon: float
    role: str


@dataclass(frozen=True)
class Trigger:
    name: str
    facility: str
    metric: str
    direction: str
    watch: float
    threshold: float
    extreme: float
    bias: int = 1
    weight: float = 1.0
    unit: str = ""
    impact: str = ""              # supply-chain mechanism, shown to the PM as the "why"
    months: tuple[int, ...] | None = None   # active season; None = year-round


@dataclass(frozen=True)
class Asset:
    key: str
    name: str
    instrument: str               # ticker the allocation is expressed in
    option_ticker: str | None     # ticker with a listed option chain (for IV / collar strikes)
    base_weight: float
    facilities: tuple[str, ...]
    triggers: tuple[Trigger, ...]
    context_tickers: tuple[str, ...] = ()
    default_stance: int = 1       # direction held when the signal is NEUTRAL
    cross_hedge: str | None = None   # set when `instrument` is a proxy for an untraded physical commodity


FACILITIES: dict[str, Facility] = {f.key: f for f in (
    Facility("donaldsonville", "Donaldsonville, LA (CF nitrogen complex)", 30.10, -90.99, "NH3 production / Gulf export"),
    Facility("haynesville", "Haynesville gas basin, LA/TX", 32.50, -93.75, "Natural gas feedstock"),
    Facility("atacama", "Salar de Atacama, Chile", -23.50, -68.30, "Lithium brine evaporation ponds"),
    Facility("antofagasta", "Antofagasta port, Chile", -23.65, -70.40, "Lithium export port"),
    Facility("florida", "Central Florida citrus belt (Polk Co.)", 28.05, -81.75, "Orange groves"),
    Facility("saopaulo", "São Paulo citrus belt (Araraquara)", -21.79, -48.18, "Orange groves / FCOJ processing"),
    Facility("santos", "Port of Santos, Brazil", -23.96, -46.33, "FCOJ bulk / cold-storage export"),
    Facility("kansas", "Western Kansas sorghum belt (Garden City)", 37.97, -100.87, "Sorghum, #1 US producing state"),
    Facility("texas_hp", "Texas High Plains (Lubbock)", 33.58, -101.85, "Dryland sorghum, #2 US producing state"),
    Facility("corpus", "Port of Corpus Christi, TX", 27.81, -97.40, "Sorghum export elevators (China demand)"),
)}

ASSETS: dict[str, Asset] = {a.key: a for a in (
    Asset(
        key="NH3",
        name="Anhydrous Ammonia",
        # No listed ammonia future on Yahoo; CF Industries is the fertilizer-equity proxy.
        # CF owns Donaldsonville, so a direct hit is two-sided (higher NH3 prices vs lost output).
        instrument="CF",
        option_ticker="CF",
        base_weight=0.25,
        facilities=("donaldsonville", "haynesville"),
        context_tickers=("NG=F", "NTR"),
        triggers=(
            Trigger("Gulf hurricane wind", "donaldsonville", "wind_speed_10m_max", ABOVE, 45, 65, 119, bias=1, unit="km/h",
                    impact="Port closures and grid outages halt NH3 synthesis and Mississippi River terminal exports."),
            Trigger("Plant freeze / power loss", "donaldsonville", "temperature_2m_min", BELOW, 0, -5, -12, bias=1, unit="°C",
                    impact="Frozen instrumentation and grid load-shedding force ammonia plant trips."),
            # A freeze-off spikes gas feedstock cost and squeezes producer margins.
            Trigger("Wellhead freeze-off", "haynesville", "temperature_2m_min", BELOW, -2, -8, -15, bias=-1, unit="°C",
                    impact="Freeze-offs cut Haynesville gas output, spiking feedstock cost for Gulf nitrogen producers."),
        ),
    ),
    Asset(
        key="LI",
        name="Lithium (brine)",
        instrument="LIT",
        option_ticker="LIT",
        base_weight=0.25,
        facilities=("atacama", "antofagasta"),
        context_tickers=("SQM", "ALB"),
        triggers=(
            Trigger("Salar flash flood / rainfall", "atacama", "precipitation_sum", ABOVE, 5, 15, 40, unit="mm",
                    impact="Rain dilutes evaporation ponds and floods access roads, setting brine concentration back months."),
            Trigger("Port wind closure", "antofagasta", "wind_speed_10m_max", ABOVE, 40, 60, 90, unit="km/h",
                    impact="High winds close Antofagasta berths, delaying lithium carbonate shipments."),
        ),
    ),
    Asset(
        key="FCOJ",
        name="Frozen Concentrated Orange Juice",
        instrument="OJ=F",
        option_ticker=None,   # ICE OJ options are not on Yahoo; IV falls back to a realized-vol proxy
        base_weight=0.25,
        facilities=("florida", "saopaulo", "santos"),
        triggers=(
            Trigger("Florida hard freeze", "florida", "temperature_2m_min", BELOW, 2, 0, -4, unit="°C",
                    impact="Sub-zero nights damage fruit and trees in the Florida citrus belt, cutting juice yield."),
            Trigger("Florida hurricane wind", "florida", "wind_speed_10m_max", ABOVE, 45, 65, 119, unit="km/h",
                    impact="Hurricane winds strip fruit and damage groves ahead of the harvest."),
            # 30-day (precipitation - reference evapotranspiration) minus its 2010-2023 seasonal normal,
            # so the normal May-Sep dry season does not read as drought.
            Trigger("São Paulo drought (water-balance anomaly)", "saopaulo", "water_balance_anom_30d", BELOW, -40, -70, -110, unit="mm",
                    impact="A rainfall deficit vs. seasonal normal stresses trees and cuts Brazil's orange crop (top FCOJ exporter)."),
            Trigger("São Paulo cyclone wind", "saopaulo", "wind_speed_10m_max", ABOVE, 45, 65, 100, unit="km/h",
                    impact="Severe convective winds cause fruit drop and processing-plant outages."),
            Trigger("Santos heavy rain (loading halts)", "santos", "precipitation_sum", ABOVE, 30, 50, 100, unit="mm",
                    impact="Heavy rain halts bulk juice loading at Santos, backing up cold-storage tanks."),
        ),
    ),
    Asset(
        key="SORG",
        name="Grain Sorghum",
        # No liquid sorghum future exists; CBOT corn is the standard cross-hedge (see the hedge tab for beta / R²).
        instrument="ZC=F",
        option_ticker="CORN",   # Teucrium Corn ETF: thin but listed chain for IV and collar strikes
        base_weight=0.25,
        facilities=("kansas", "texas_hp", "corpus"),
        context_tickers=("CORN",),
        cross_hedge="Sorghum has no liquid futures contract; exposure and hedges run through CBOT corn (ZC=F).",
        triggers=(
            Trigger("Kansas flowering heat", "kansas", "temperature_2m_max", ABOVE, 35, 38, 42, unit="°C", months=(6, 7, 8, 9),
                    impact="Heat at flowering and grain fill blasts sorghum heads in western Kansas, the #1 producing state."),
            Trigger("Kansas drought (water-balance anomaly)", "kansas", "water_balance_anom_30d", BELOW, -40, -70, -110,
                    unit="mm", months=(5, 6, 7, 8, 9),
                    impact="A growing-season rain deficit vs. normal cuts Kansas dryland sorghum yields."),
            Trigger("Texas High Plains heat", "texas_hp", "temperature_2m_max", ABOVE, 36, 39, 43, unit="°C", months=(6, 7, 8, 9),
                    impact="Extreme heat on the Texas High Plains stresses dryland sorghum during grain fill."),
            Trigger("Texas Gulf hurricane (export elevators)", "corpus", "wind_speed_10m_max", ABOVE, 45, 65, 119,
                    unit="km/h", months=(6, 7, 8, 9, 10, 11),
                    impact="Hurricane closures at Corpus Christi / Houston elevators halt sorghum exports to China."),
        ),
    ),
)}

# --- Risk engine -------------------------------------------------------------
LEAD_DECAY = 0.92            # per forecast day: less trust in day-7 than day-1
LOOKBACK_OBSERVED_DAYS = 1   # also score yesterday's observed weather
FORECAST_DAYS = 8
PAST_DAYS = 31               # enough history for the 30-day water balance
CLIMATOLOGY_YEARS = (2010, 2023)   # seasonal-normal baseline; ends before the backtest window
ENSEMBLE_MODELS = ("gfs_seamless", "ecmwf_ifs025")   # pooled members -> breach probabilities
CONFIDENCE_LEVELS = (("High", 0.80), ("Medium", 0.60), ("Low", 0.0))   # share of members agreeing
MIN_SIGNAL = 0.25            # below this C_event the bias is NEUTRAL

# --- Sizing ------------------------------------------------------------------
VOL_WINDOW = 14
VOL_SCALE_BOUNDS = (0.25, 3.0)

# --- Hedging / capital preservation -----------------------------------------
COLLAR_CONFIDENCE = 0.75
IV_SPIKE_THRESHOLD = 0.30    # implied vol at least 30% above 30-day realized vol
COLLAR_PUT_MONEYNESS = 0.95
VAR_CONFIDENCE = 0.95
VAR_LOOKBACK = 250
VAR_LIMIT = 0.03             # 1-day 95% VaR as a fraction of NAV
CIRCUIT_BREAKER_CUT = 0.50
