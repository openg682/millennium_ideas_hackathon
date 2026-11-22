# Architecture

```
            Open-Meteo /v1/forecast            Yahoo Finance (yfinance)
     (1 batched call, 7 facilities, 31d past   (CF, LIT, OJ=F, NG=F, NTR, SQM, ALB
      + 8d forecast, no API key)                + CF/LIT near-month option chains)
                  │                                        │
                  └──────────── ingestion.py ──────────────┘
                                     │  WeatherSnapshot / MarketSnapshot
                                     ▼
                     pipeline.py  (+ optional stress scenario overrides)
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                             ▼
   signals.py                    sizing.py                     hedging.py
   severity ramp per trigger     Base × VolScale ×             Zero-cost collar alert
   × lead-time decay →           (0.5 + 1.25·C_event)          (C ≥ 0.75 & IV/RV30−1 > 30%)
   noisy-OR → C_event, bias      VolScale = inverse 14d vol    Historical VaR circuit breaker
        └────────────────────────────┼─────────────────────────────┘
                                     ▼
                              PortfolioReport (models.py)
                         ┌───────────┴────────────┐
                 python -m millenium        streamlit run dashboard/app.py
```

| Module | Responsibility |
|---|---|
| `millenium/config.py` | Facilities, triggers (watch / threshold / extreme), instruments, risk limits |
| `millenium/ingestion.py` | Open-Meteo + yfinance clients, derived 30d water balance, ATM IV, realized vol |
| `millenium/signals.py` | `RiskEngine`: severity ramps → Event Confidence Score + LONG/SHORT/NEUTRAL |
| `millenium/sizing.py` | `PositionSizer`: inverse-vol scaling × conviction multiplier |
| `millenium/hedging.py` | `HedgeManager`: collar strikes from the live chain, VaR circuit breaker |
| `millenium/probability.py` | `EnsembleAnalyzer`: re-runs the engine on every ensemble member → P(breach), C_event spread |
| `millenium/advisor.py` | `Advisor`: action vs the PM's weights, evidence-based reasons, confidence level |
| `millenium/backtest.py` | `Backtester`: PM weights held static vs AI-resized, no look-ahead |
| `millenium/pipeline.py` | Orchestration + demo stress scenarios |
| `dashboard/app.py` | Tabs: Desk (PM ‖ AI), Facility weather, Disruption probabilities, Sorghum ↔ corn hedge, Benchmark vs AI |

## Data sources (all free, no keys)
| Source | Used for |
|---|---|
| Open-Meteo forecast API | Deterministic 8-day forecast + 31 observed days per facility |
| Open-Meteo ensemble API (GFS 31 + ECMWF 51 members) | Breach probabilities, forecast bands, recommendation confidence |
| Open-Meteo archive API (ERA5) | 2010–2023 seasonal normals (drought anomaly) and 2-year backtest weather |
| Yahoo Finance via yfinance | Prices, realized vol, CF/LIT/CORN option chains (IV, collar strikes), ZC=F history |
| USDA ERS Feed Grains Database (bulk CSV) | Monthly sorghum & corn cash / farm prices for the sorghum–corn cross-hedge |

## Sorghum and the corn cross-hedge
Sorghum has no liquid futures contract, so the SORG sleeve is held in CBOT corn (ZC=F) and hedged with it
(collar strikes from the CORN ETF chain). `hedging.cross_hedge_stats` regresses monthly log changes of USDA
sorghum prices on monthly-average ZC=F: β is the hedge ratio, R² the share of price risk corn removes, and the
residual is basis risk. Central Kansas cash (2020–) is the primary estimate; US farm price (2000–) is the long-run check.
Sorghum triggers are seasonal (`Trigger.months`): heat Jun–Sep, drought May–Sep, Texas Gulf hurricanes Jun–Nov.

## Confidence
Each ensemble member is pushed through the same severity → C_event → sizing chain. Confidence is the share
of members whose weather leads to the same action as the deterministic forecast (High ≥ 80%, Medium ≥ 60%).

## Scoring
- **Severity**: `watch → 0`, `threshold → 0.5`, `extreme → 1.0` (linear in between), discounted `0.92^lead_days`,
  peak taken over yesterday + 8 forecast days.
- **C_event**: noisy-OR across an asset's triggers, `1 − Π(1 − wᵢ·sᵢ)`.
- **Bias**: sign of `Σ biasᵢ·sᵢ`; NEUTRAL below `C_event = 0.25` (position then holds its default long stance).
- **IV spike**: ATM implied vol / 30d realized − 1. OJ=F has no Yahoo option chain, so it uses 10d realized vol as the proxy.
- **VaR**: 1-day 95% historical simulation over 250 days of the signed instrument weights; > 3% NAV cuts all weights 50%.

## Known PoC limitations
- Trigger directions (e.g. hurricane → LONG CF, freeze-off → SHORT CF) are hypotheses; backtest before use.
- The backtest scores observed weather (no archived forecasts), so it understates the lead time live mode gets.
- Santos cold-storage congestion is not weather-observable; it needs port/AIS data.
