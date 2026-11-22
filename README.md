You are an expert quantitative developer and full-stack engineer. I want to build a Proof-of-Concept (PoC) dashboard and back-end pipeline for an AI-driven Niche Commodity Trading & Risk Management Application.

### Project Context & Objective
The app monitors physical supply disruptions caused by natural disasters, severe weather, and logistics bottlenecks across three specific niche commodities with highly concentrated, physically observable supply chains:

1. Anhydrous Ammonia (NH3)
   - Primary Facilities: Gulf Coast plants (e.g., Donaldsonville, LA; Lat 30.10, Lon -90.99) & nitrogen complexes.
   - Core Triggers: Gulf hurricanes (port closures/power outages) and severe winter freeze-offs (nat gas feedstock spikes & wellhead freezes).
   - Market Proxies: Fertilizer equities / OTC Ammonia index metrics.

2. Lithium Carbonate/Hydroxide (Brine-Based)
   - Primary Facilities: Atacama Salt Flat, Chile (Lat -23.50, Lon -68.30) & regional export ports (Antofagasta).
   - Core Triggers: Rare unseasonal rainfall / flash flooding in high-altitude salt flats (disrupting solar evaporation cycles).
   - Market Proxies: Global X Lithium & Battery Tech ETF (LIT) / Lithium spot proxies.

3. Frozen Concentrated Orange Juice (FCOJ)
   - Primary Facilities: Central Florida citrus belt & São Paulo state, Brazil.
   - Core Triggers: Hard freezes in Florida, severe drought/cyclones in Brazil, port cold-storage congestion.
   - Market Proxies: ICE FCOJ Futures (OJ=F).

---

### Core System Modules to Implement

1. Data Ingestion Module:
   - Connect to Open-Meteo API (free REST weather API) for targeted coordinates (Temperature, Precipitation, Wind Speed).
   - Fetch asset prices via `yfinance` (`OJ=F`, `LIT`, and relevant proxies/ETFs).

2. Risk Engine & Signal Generation:
   - Define disaster/disruption thresholds for each asset (e.g., Wind > 65 km/h for Ammonia, Rainfall > 15mm/day for Lithium brine, Temp < 0°C for FCOJ).
   - Output an Event Confidence Score (C_event from 0.0 to 1.0) and directional bias (LONG / SHORT / NEUTRAL).

3. Dynamic Position Resizing Engine:
   - Scale position size dynamically using:
     Effective Size = Base Weight * Volatility Scale * (0.5 + 1.25 * C_event)
   - Adjust weights inversely to 14-day historical volatility.

4. Hedging & Capital Preservation Provisions:
   - Trigger a "Zero-Cost Collar" alert if C_event >= 0.75 and Implied Volatility spikes > 30%.
   - Trigger a "Circuit Breaker" to cut allocations by 50% if portfolio VaR exceeds a safety threshold.

---

### Your First Task
Please provide:
1. System Architecture Diagram / Code Structure: Outline the modular Python project layout.
2. Production-Ready Python Backend: A clean, object-oriented script that ingests the live weather and market data, calculates the risk metrics, resizes allocations, and triggers hedging rules for all 3 commodities.
3. Interactive Dashboard Concept: Code for a frontend dashboard (e.g., Streamlit or Plotly Dash) that displays:
   - Real-time weather risk map/cards for each facility coordinate.
   - Live portfolio allocation breakdown & risk weights.
   - Active execution signals and automated hedging alerts.

Let's start by generating the complete Python backend script and dashboard setup.