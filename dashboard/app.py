"""Streamlit dashboard: `streamlit run dashboard/app.py` from the project root.

Built for a portfolio manager: every tab leads with one decision-level view; detail sits in expanders.
"""
from __future__ import annotations

import html
import math
import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from millenium import config  # noqa: E402
from millenium.advisor import Advisor  # noqa: E402
from millenium.backtest import Backtester, signal_hit_rates  # noqa: E402
from millenium.eod import claude_note, eod_facts, rule_based_note  # noqa: E402
from millenium.hedging import historical_var  # noqa: E402
from millenium.pipeline import SCENARIOS, Pipeline, apply_weather_scenario  # noqa: E402
from millenium.supplychain import (BREACH, CRITICAL, LEVEL_NAME, NORMAL, UNKNOWN, WATCH,  # noqa: E402
                                   SupplyChainMonitor, evaluate_flow)

st.set_page_config(page_title="Millenium Commodity Risk", page_icon="🌀", layout="wide")

# --- Palette (dataviz reference instance; dark steps chosen, not flipped) -------------------------
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
SURFACE = "#1a1a19" if DARK else "#fcfcfb"
INK = "#f0efec" if DARK else "#0b0b0b"
MUTED = "#c3c2b7" if DARK else "#52514e"
GRID = "rgba(128,128,128,0.18)"
ASSET_COLOR = ({"NH3": "#3987e5", "LI": "#d95926", "FCOJ": "#199e70", "SORG": "#c98500"} if DARK
               else {"NH3": "#2a78d6", "LI": "#eb6834", "FCOJ": "#1baf7a", "SORG": "#eda100"})
HEDGE_COLOR = "#d55181" if DARK else "#e87ba4"
PM_COLOR, AI_COLOR = "#8f8e88", ("#9085e9" if DARK else "#6250d6")
LEVEL_COLOR = {UNKNOWN: "#8f8e88", NORMAL: "#0ca30c", WATCH: "#fab219", BREACH: "#ec835a", CRITICAL: "#d03b3b"}
LEVEL_ICON = {UNKNOWN: "⚪", NORMAL: "🟢", WATCH: "🟡", BREACH: "🟠", CRITICAL: "🔴"}
LEVEL_TINT = ({UNKNOWN: "#2a2a28", NORMAL: "#1d2a1d", WATCH: "#352c12", BREACH: "#3a261d", CRITICAL: "#3b1f1f"} if DARK
              else {UNKNOWN: "#f1f1ef", NORMAL: "#eef7ee", WATCH: "#fff6df", BREACH: "#fdebe3", CRITICAL: "#fbe3e3"})
SEQ = ([[0, "#232322"], [0.001, "#104281"], [0.35, "#256abf"], [0.7, "#5598e7"], [1, "#cde2fb"]] if DARK
       else [[0, "#f3f5f8"], [0.001, "#cde2fb"], [0.35, "#86b6ef"], [0.7, "#2a78d6"], [1, "#0d366b"]])
ACTION_ICON = {"INCREASE": "🔼", "REDUCE": "🔽", "HOLD": "⏸", "FLIP TO LONG": "🔁", "FLIP TO SHORT": "🔁"}
CONF_ICON = {"High": "●●●", "Medium": "●●○", "Low": "●○○", "n/a": "○○○"}


def style(fig: go.Figure, height: int, **layout) -> go.Figure:
    base = dict(margin=dict(l=0, r=10, t=20, b=0), hoverlabel=dict(namelength=-1), legend=dict(orientation="h", y=-0.2))
    fig.update_layout(height=height, **(base | layout))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def short(fac_key: str) -> str:
    return config.FACILITIES[fac_key].name.split(" (")[0]


def name(asset_key: str) -> str:
    return config.ASSETS[asset_key].name


# --- Data (cached) ---------------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner="Loading forecasts, ensembles and market prices…")
def load_snapshots():
    return Pipeline().fetch()


@st.cache_data(ttl=1800, show_spinner="Checking ports, rivers, sea lanes and disaster alerts…")
def load_supply(today_key: str, _prices):
    return SupplyChainMonitor().fetch(_prices)


@st.cache_data(ttl=24 * 3600, show_spinner="Loading physical sorghum and orange prices…")
def load_cross_hedges():
    return Pipeline().fetch_cross_hedges()


@st.cache_data(ttl=6 * 3600, show_spinner="Loading observed weather history…")
def load_history(start, today_key: str, _weather):
    return Pipeline().fetch_weather_history(_weather, start)


@st.cache_data(ttl=6 * 3600, show_spinner="Replaying the AI over history…")
def run_backtest(weights: tuple, today_key: str, _history, _prices):
    return Backtester().run(_history, _prices, dict(weights))


with st.sidebar:
    st.header("Controls")
    scenario = st.selectbox("Scenario", list(SCENARIOS),
                            help="Stress scenarios override forecast days +1/+2 to show how alerts behave.")
    nav = st.number_input("Portfolio NAV ($)", min_value=100_000, value=10_000_000, step=500_000, format="%d")
    if st.button("Refresh live data"):
        load_snapshots.clear()
        load_supply.clear()
    with st.expander("Rules"):
        st.caption(f"Size = your weight × inverse-vol scale × (0.5 + 1.25·C_event)  \n"
                   f"Collar alert: C_event ≥ {config.COLLAR_CONFIDENCE} and IV > 30d realized by {config.IV_SPIKE_THRESHOLD:.0%}  \n"
                   f"Circuit breaker: book VaR > {config.VAR_LIMIT:.0%} → cut {config.CIRCUIT_BREAKER_CUT:.0%}  \n"
                   f"Confidence: share of 82 forecast scenarios that lead to the same action")

weather, market = load_snapshots()
today_key = str(max(weather.today.values()))
supply = load_supply(today_key, market.prices)
cross = load_cross_hedges()
if "pm_weights" not in st.session_state:
    st.session_state.pm_weights = {k: round(a.base_weight, 3) for k, a in config.ASSETS.items()}

st.title("Commodity Disruption Desk")
banner = st.empty()
tab_today, tab_supply, tab_hedge, tab_perf, tab_eod = st.tabs(
    ["🧭 Today", "⛓ Supply chain", "🛡 Hedges", "📊 Performance", "📝 End of day"])

# ==================================================================================================
# Today: your book (left) and the AI's recommendations (right)
# ==================================================================================================
with tab_today:
    pm_col, ai_col = st.columns([1, 1.3], gap="large")
    with pm_col:
        st.subheader("Your book")
        px = market.prices
        last = {k: px[a.instrument].dropna() for k, a in config.ASSETS.items()}
        editor = st.data_editor(
            pd.DataFrame([{"Asset": a.name, "Instrument": a.instrument, "Last": last[k].iloc[-1],
                           "1D": last[k].iloc[-1] / last[k].iloc[-2] - 1, "Weight": st.session_state.pm_weights[k]}
                          for k, a in config.ASSETS.items()]),
            hide_index=True, width="stretch", disabled=["Asset", "Instrument", "Last", "1D"], key="pm_editor",
            column_config={"Weight": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                                                                   help="Fraction of NAV. Edit to test your own book."),
                           "Last": st.column_config.NumberColumn(format="%.2f"),
                           "1D": st.column_config.NumberColumn(format="percent")})
        pm_weights = {k: float(w) for k, w in zip(config.ASSETS, editor["Weight"].fillna(0))}
        st.session_state.pm_weights = pm_weights

    report = Pipeline().evaluate(weather, market, scenario, base_weights=pm_weights)
    flows = {k: evaluate_flow(k, report, supply) for k in config.ASSETS}
    evidence = {k: [f"{i.name} — {i.reading}" for sr in stages for i in sr.items if i.kind == "feed" and i.level >= WATCH]
                for k, stages in flows.items()}
    recs = Advisor().recommend(report, pm_weights, weather.today, cross, evidence)
    returns = market.prices.pct_change(fill_method=None)
    pm_var = historical_var(returns, {config.ASSETS[k].instrument: w for k, w in pm_weights.items()})

    with pm_col:
        day_pnl = sum(pm_weights[k] * (last[k].iloc[-1] / last[k].iloc[-2] - 1) * nav for k in config.ASSETS)
        m1, m2, m3 = st.columns(3)
        m1.metric("Day P&L", f"${day_pnl:,.0f}")
        m2.metric("Book VaR (1d, 95%)", f"{pm_var:.2%}", delta=f"limit {config.VAR_LIMIT:.0%}", delta_color="off")
        m3.metric("Invested", f"{sum(pm_weights.values()):.0%}")
        if pm_var > config.VAR_LIMIT:
            st.warning(f"Your book's VaR is above the {config.VAR_LIMIT:.0%} limit.")

    with ai_col:
        st.subheader("AI recommendations")
        st.dataframe(pd.DataFrame([{
            "Asset": name(r.asset), "Action": f"{ACTION_ICON.get(r.action, '')} {r.action}",
            "Your → AI weight": f"{r.pm_weight:.3f} → {r.ai_weight:+.3f}",
            "Confidence": f"{CONF_ICON[r.confidence_level]} {r.confidence_level}", "Why": r.headline,
        } for r in recs]), hide_index=True, width="stretch",
            column_config={"Why": st.column_config.TextColumn(width="large")})
        for alert in report.alerts:
            (st.error if alert.kind == "CIRCUIT_BREAKER" else st.warning)(
                f"**{alert.kind.replace('_', ' ').title()}** — {alert.message}")
        with st.expander("Why? Reasoning per asset"):
            for r in recs:
                st.markdown(f"**{name(r.asset)}** — {r.action}, confidence {r.confidence_level}"
                            + (f" ({r.confidence:.0%} of forecast scenarios agree)" if r.confidence is not None else ""))
                st.markdown("\n".join(f"- {x}" for x in r.reasons))

# ==================================================================================================
# Supply chain: one flow per commodity, flagged risks first
# ==================================================================================================
def flow_dot(stages) -> str:
    def cell(text, size=10, color=INK, bold=False):
        t = html.escape(text)
        return f'<FONT POINT-SIZE="{size}" COLOR="{color}">{"<B>" + t + "</B>" if bold else t}</FONT>'
    lines = ["digraph flow {", 'graph [rankdir=LR bgcolor="transparent" nodesep=0.25 ranksep=0.4 pad=0.1];',
             'node [shape=plain fontname="Helvetica"];', f'edge [color="{MUTED}" penwidth=1.6 arrowsize=0.7];']
    for sr in stages:
        title, _, place = sr.stage.label.partition(" · ")
        flagged = [i for i in sr.items if i.level >= WATCH]
        quiet = sum(1 for i in sr.items if i.level == NORMAL)
        rows = [f'<TR><TD ALIGN="LEFT">{cell(title, 11, bold=True)}</TD></TR>']
        if place:
            rows.append(f'<TR><TD ALIGN="LEFT">{cell(place, 9, MUTED)}</TD></TR>')
        rows.append(f'<TR><TD ALIGN="LEFT"><FONT COLOR="{LEVEL_COLOR[sr.level]}">●</FONT> {cell(LEVEL_NAME[sr.level], 10, bold=True)}</TD></TR>')
        for i in flagged[:2]:
            rows.append(f'<TR><TD ALIGN="LEFT"><FONT COLOR="{LEVEL_COLOR[i.level]}">●</FONT> {cell(i.name[:34], 9)}</TD></TR>')
        if sr.items and not flagged:
            plural = "s" if quiet != 1 else ""
            rows.append(f'<TR><TD ALIGN="LEFT">{cell(f"{quiet} check{plural} normal", 9, MUTED)}</TD></TR>')
        table = (f'<TABLE BORDER="2" COLOR="{LEVEL_COLOR[sr.level]}" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" '
                 f'BGCOLOR="{LEVEL_TINT[sr.level]}">{"".join(rows)}</TABLE>')
        lines.append(f'"{sr.stage.key}" [label=<{table}>];')
    lines += [f'"{up}" -> "{sr.stage.key}";' for sr in stages for up in sr.stage.upstream]
    return "\n".join(lines + ["}"])


with tab_supply:
    tiles = st.columns(len(config.ASSETS))
    for col, (k, stages) in zip(tiles, flows.items()):
        worst = max((sr.level for sr in stages if sr.level != UNKNOWN), default=NORMAL)
        ap = report.asset_probabilities.get(k)
        col.metric(name(k), f"{LEVEL_ICON[worst]} {LEVEL_NAME[worst]}",
                   delta=f"{ap.p_any:.0%} chance of a weather disruption in 8 days" if ap else None, delta_color="off")
    chosen = st.radio("Commodity", list(config.ASSETS), format_func=name, horizontal=True, label_visibility="collapsed")
    stages = flows[chosen]
    st.graphviz_chart(flow_dot(stages), width="stretch")
    st.caption("🟢 Normal  🟡 Watch  🟠 Breach  🔴 Critical  ⚪ Not connected — each box is one stage of the physical chain; "
               "it takes the worst status of the checks attached to it.")

    flagged = [(sr, i) for sr in stages for i in sr.items if i.level >= WATCH]
    if flagged:
        st.markdown("**What could break**")
        for sr, i in sorted(flagged, key=lambda x: -x[1].level):
            st.markdown(f"{LEVEL_ICON[i.level]} **{sr.stage.label}** — {i.name}: {i.reading}"
                        + (f"  \n<small>{html.escape(i.why)}</small>" if i.why else ""), unsafe_allow_html=True)
    else:
        st.success("Every stage of this chain is normal right now.")

    with st.expander("All checks for this chain"):
        st.dataframe(pd.DataFrame([{"Stage": sr.stage.label, "Check": i.name, "Status": f"{LEVEL_ICON[i.level]} {LEVEL_NAME[i.level]}",
                                    "Reading": i.reading, "Source": i.source, "Why it matters": i.why}
                                   for sr in stages for i in sr.items]), hide_index=True, width="stretch")

    with st.expander("Weather at the facilities"):
        daily = apply_weather_scenario(weather.daily, weather.today, SCENARIOS[scenario]["weather"], weather.climatology)
        trig_prob = {(t.asset, t.trigger): t for t in report.trigger_probabilities}
        fac_key = st.selectbox("Facility", config.ASSETS[chosen].facilities, format_func=short)
        today = weather.today[fac_key]
        frame = daily[fac_key]
        view = frame.loc[(frame.index >= today - timedelta(days=14)) & (frame.index <= today + timedelta(days=config.FORECAST_DAYS))]
        for trig in [t for t in config.ASSETS[chosen].triggers if t.facility == fac_key]:
            tp = trig_prob.get((chosen, trig.name))
            obs, fc = view.loc[view.index <= today, trig.metric], view.loc[view.index >= today, trig.metric]
            fig = go.Figure()
            if tp is not None and tp.band is not None:
                band = tp.band.loc[tp.band.index >= today]
                fig.add_trace(go.Scatter(x=list(band.index) + list(band.index[::-1]), y=list(band["p90"]) + list(band["p10"][::-1]),
                                         fill="toself", fillcolor="rgba(42,120,214,0.18)", line=dict(width=0),
                                         name="Forecast range (p10–p90)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=obs.index, y=obs, name="Observed", mode="lines+markers",
                                     line=dict(width=2, color=ASSET_COLOR["NH3"]), marker=dict(size=8), hovertemplate="%{y:.1f}"))
            fig.add_trace(go.Scatter(x=fc.index, y=fc, name="Forecast", mode="lines+markers",
                                     line=dict(width=2, color=ASSET_COLOR["NH3"], dash="dash"), marker=dict(size=8),
                                     hovertemplate="%{y:.1f}"))
            fig.add_hline(y=trig.threshold, line=dict(color=LEVEL_COLOR[CRITICAL], width=1.5, dash="dash"),
                          annotation_text=f"Trigger {trig.threshold:g} {trig.unit}", annotation_font_color=MUTED)
            fig.add_vline(x=today, line=dict(color=MUTED, width=1))
            st.markdown(f"**{trig.name}**" + (f" · {tp.p_breach:.0%} chance in 8 days" if tp else ""))
            st.plotly_chart(style(fig, 220, hovermode="x unified", yaxis_title=trig.unit), width="stretch",
                            key=f"wx-{fac_key}-{trig.name}")

    with st.expander("Breach probability by day (all commodities)"):
        ranked = sorted(report.trigger_probabilities, key=lambda t: -t.p_breach)
        if ranked:
            days = sorted({d for t in ranked for d in t.p_by_day})
            z = [[t.p_by_day.get(d) for d in days] for t in ranked]
            fig = go.Figure(go.Heatmap(z=z, x=[d.strftime("%a %d") for d in days], y=[f"{short(t.facility)} · {t.trigger}" for t in ranked],
                                       text=[[f"{v:.0%}" if v else "" for v in row] for row in z], texttemplate="%{text}",
                                       colorscale=SEQ, zmin=0, zmax=1, xgap=2, ygap=2,
                                       colorbar=dict(title="P(breach)", tickformat=".0%"),
                                       hovertemplate="%{y}<br>%{x}: %{z:.0%}<extra></extra>"))
            st.plotly_chart(style(fig, 60 + 26 * len(ranked), yaxis=dict(autorange="reversed")), width="stretch")

    with st.expander("Latest news"):
        articles = supply.news.get(chosen, [])
        if articles:
            st.markdown("\n".join(f"- [{a['title']}]({a['url']}) · {a['source']}" for a in articles))
        else:
            gdelt = next((f for f in supply.feeds if f.name.startswith("GDELT")), None)
            st.caption(f"No headlines: {gdelt.message if gdelt and not gdelt.ok else 'nothing matched in the last 3 days'}.")

# ==================================================================================================
# Hedges: physical commodity vs the future you can trade
# ==================================================================================================
with tab_hedge:
    options = {k: v for k, v in cross.items() if "stats" in v}
    missing = {k: v["error"] for k, v in cross.items() if "error" in v}
    for k, err in missing.items():
        st.info(f"{name(k)} hedge data unavailable: {err}")
    if options:
        key = st.radio("Hedge", list(options), format_func=lambda k: options[k]["title"], horizontal=True,
                       label_visibility="collapsed")
        b = options[key]
        label, hs = next(iter(b["stats"].items()))
        st.caption(f"Primary estimate: {label}, {hs['start']:%b %Y} – {hs['end']:%b %Y}, monthly. "
                   f"Physical {b['physical']} has no listed future of its own, so it is hedged with {b['hedge']}.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Hedge ratio", f"{hs['beta']:.2f}", help=f"$ of {b['hedge']} to sell per $1 of physical {b['physical']}")
        c2.metric("Price risk removed", f"{hs['r2']:.0%}", help="R² of monthly price changes")
        c3.metric("Left unhedged (basis)", f"{hs['basis_vol']:.0%} / yr", help=f"{hs['spot_vol']:.0%} / yr with no hedge")

        q1, q2, q3 = st.columns(3)
        qty = q1.number_input(f"Physical exposure ({b['qty_unit']})", min_value=0, value=b["qty_default"],
                              step=max(1, b["qty_default"] // 20), key=f"qty-{key}")
        long_physical = q2.radio("Position", ["Long (own / grow)", "Short (forward sold)"], horizontal=True,
                                 key=f"pos-{key}").startswith("Long")
        hedge_px = float(market.prices[b["hedge"]].dropna().iloc[-1]) / 100
        value = qty * b["unit_price"]
        contracts = hs["beta"] * value / (hedge_px * b["contract_size"])
        side = "Sell" if long_physical else "Buy"
        q3.metric(f"{side} {b['hedge']}", f"{contracts:,.0f} contracts", help=f"{b['contract_size']:,} {b['contract_unit']} per contract")
        z = 1.645 / math.sqrt(12)
        st.markdown(f"About **${value:,.0f}** of {b['physical']} → {side.lower()} **{contracts:,.0f}** {b['hedge']} contracts. "
                    f"One-month 95% price risk drops from **${z * hs['spot_vol'] * value:,.0f}** to "
                    f"**${z * hs['basis_vol'] * value:,.0f}**.")

        with st.expander("Charts and second estimate"):
            fig = go.Figure()
            for (lbl, s), dash in zip(b["series"].items(), ("solid", "dot")):
                s = s.dropna().loc["2015":]
                fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, name=lbl, mode="lines",
                                         line=dict(width=2, color=ASSET_COLOR[key], dash=dash), hovertemplate="%{y:.0f}"))
            h = b["hedge_series"].dropna().loc["2015":]
            fig.add_trace(go.Scatter(x=h.index, y=h / h.iloc[0] * 100, name=f"{b['hedge']} (monthly avg)", mode="lines",
                                     line=dict(width=2, color=HEDGE_COLOR), hovertemplate="%{y:.0f}"))
            st.markdown("**Prices indexed to 100 (2015)**")
            st.plotly_chart(style(fig, 300, hovermode="x unified"), width="stretch")
            st.dataframe(pd.DataFrame([{"Estimate": lbl, "Months": s["n_months"], "Hedge ratio": s["beta"],
                                        "Risk removed": s["r2"], "Basis vol": s["basis_vol"]} for lbl, s in b["stats"].items()]),
                         hide_index=True, width="stretch", column_config={
                             "Hedge ratio": st.column_config.NumberColumn(format="%.2f"),
                             "Risk removed": st.column_config.NumberColumn(format="percent"),
                             "Basis vol": st.column_config.NumberColumn(format="percent")})
            if key == "FCOJ":
                st.caption("Both orange series are fresh-market prices (no free juice-processor price series exists); "
                           "a processor's own delivered-in price would give a tighter hedge estimate.")

# ==================================================================================================
# Performance: your weights held static vs the AI's resizing
# ==================================================================================================
history = load_history(market.prices.index.min().date(), today_key, weather)
bt = run_backtest(tuple(sorted(pm_weights.items())), today_key, history, market.prices)
with tab_perf:
    m = bt.metrics
    b_row, a_row = m.loc["Benchmark (PM static)"], m.loc["AI resized"]
    st.caption(f"{bt.returns.index[0]:%b %Y} – {bt.returns.index[-1]:%b %Y} · benchmark = your weights held constant · "
               "AI = same weights resized daily from weather seen up to the day before, after 5 bps costs.")
    p1, p2, p3 = st.columns(3)
    p1.metric("Total return (AI vs yours)", f"{a_row['Total return']:+.1%}", delta=f"{a_row['Total return'] - b_row['Total return']:+.1%}")
    p2.metric("Sharpe (AI vs yours)", f"{a_row['Sharpe']:.2f}", delta=f"{a_row['Sharpe'] - b_row['Sharpe']:+.2f}")
    p3.metric("Worst drawdown (AI vs yours)", f"{a_row['Max drawdown']:.0%}", delta=f"{a_row['Max drawdown'] - b_row['Max drawdown']:+.0%}")
    fig = go.Figure()
    for s, color in (("Benchmark (PM static)", PM_COLOR), ("AI resized", AI_COLOR)):
        eq = bt.equity[s] * 100
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Your weights (static)" if s.startswith("Bench") else "AI resized",
                                 mode="lines", line=dict(width=2, color=color), hovertemplate="%{y:.1f}"))
        fig.add_annotation(x=eq.index[-1], y=eq.iloc[-1], text=f"{eq.iloc[-1]:.0f}", showarrow=False, xanchor="left",
                           font=dict(size=11, color=MUTED))
    st.plotly_chart(style(fig, 340, hovermode="x unified", yaxis_title="Growth of 100"), width="stretch")
    st.caption(f"The AI averaged {a_row['Avg gross']:.2f}× exposure vs your {b_row['Avg gross']:.2f}×, so part of any drawdown "
               "difference is simply lower exposure; the vol-matched row in the details compares timing at equal risk.")
    with st.expander("Details"):
        st.dataframe(m.reset_index(), hide_index=True, width="stretch", column_config={
            **{c: st.column_config.NumberColumn(format="percent") for c in ("Total return", "Ann. return", "Ann. vol", "Max drawdown")},
            "Sharpe": st.column_config.NumberColumn(format="%.2f"), "Avg gross": st.column_config.NumberColumn(format="%.2f×"),
            "Circuit breaker days": st.column_config.NumberColumn(format="%d")})
        fig = go.Figure()
        for k in bt.c_event:
            fig.add_trace(go.Scatter(x=bt.c_event.index, y=bt.c_event[k], name=name(k), mode="lines",
                                     line=dict(width=2, color=ASSET_COLOR[k]), hovertemplate="%{y:.2f}"))
        st.markdown("**When the AI saw disruptions (C_event)**")
        st.plotly_chart(style(fig, 260, hovermode="x unified", yaxis=dict(range=[0, 1.02])), width="stretch")

# ==================================================================================================
# End of day: what worked, what failed, what to watch
# ==================================================================================================
data_status = [
    {"name": "Open-Meteo forecast", "provides": "8-day forecast + 31 observed days, 10 facilities", "ok": bool(weather.daily),
     "as_of": today_key, "message": "ok"},
    {"name": "Open-Meteo ensemble", "provides": "82 forecast scenarios → probabilities & confidence",
     "ok": bool(weather.ensemble_models), "as_of": today_key, "message": ", ".join(weather.ensemble_models) or "unavailable"},
    {"name": "Open-Meteo archive", "provides": "2010–2023 seasonal normals, 2-year backtest weather", "ok": bool(history),
     "as_of": today_key, "message": "ok"},
    {"name": "Yahoo Finance", "provides": "Prices for CF, LIT, OJ=F, ZC=F, NG=F + option chains (CF, LIT, CORN)",
     "ok": not market.prices.empty, "as_of": str(market.prices.index[-1].date()),
     "message": f"option chains: {', '.join(market.option_chains) or 'none'}"},
    {"name": "USDA ERS Feed Grains", "provides": "Monthly sorghum cash & farm prices", "ok": "stats" in cross.get("SORG", {}),
     "as_of": str(cross.get("SORG", {}).get("unit_price_as_of", ""))[:10], "message": cross.get("SORG", {}).get("error", "ok")},
    {"name": "World Bank Pink Sheet + BLS", "provides": "Monthly orange prices (EU import, US retail Navel)",
     "ok": "stats" in cross.get("FCOJ", {}), "as_of": str(cross.get("FCOJ", {}).get("unit_price_as_of", ""))[:10],
     "message": cross.get("FCOJ", {}).get("error", "ok")},
    *[{"name": f.name, "provides": f.provides, "ok": f.ok, "as_of": f.as_of, "message": f.message} for f in supply.feeds],
]
with tab_eod:
    facts = eod_facts(report, recs, bt, supply, pm_weights, nav, data_status)
    note = rule_based_note(facts)
    st.caption(f"Session {facts['date']} · last prices · AI weights are those it held into today (weather up to yesterday).")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Your book today", f"${facts['day_total']['pm_pnl']:,.0f}")
    e2.metric("AI-resized today", f"${facts['day_total']['ai_pnl']:,.0f}")
    e3.metric("AI vs you today", f"${facts['day_total']['ai_pnl'] - facts['day_total']['pm_pnl']:+,.0f}")
    e4.metric("AI vs you, 5 days", f"{facts['week_total']['ai_return'] - facts['week_total']['pm_return']:+.2%}")
    w, f, t = st.columns(3)
    w.markdown("**✅ What worked**\n" + "\n".join(f"- {x}" for x in note["worked"]))
    f.markdown("**❌ What failed**\n" + "\n".join(f"- {x}" for x in note["failed"]))
    t.markdown("**👀 Watch tomorrow**\n" + "\n".join(f"- {x}" for x in note["watch"]))

    st.divider()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if st.button("✍️ Write today's note with Claude", help="Sends the facts above (no positions beyond weights) to Claude Opus 5.5."):
        st.session_state.claude_note = claude_note(facts)
    text, problem = st.session_state.get("claude_note", (None, None))
    if text:
        st.markdown(text)
    elif problem:
        st.info(f"{problem} The rule-based summary above is complete without it.")
    elif not has_key:
        st.caption("Optional: set ANTHROPIC_API_KEY to have Claude write the note in prose.")

    with st.expander("Signal scorecard (backtest)"):
        sc = signal_hit_rates(bt).rename(index=name).rename(columns={
            "signal_days": "Signal days", "hit_rate": "Right after 5 days", "avg_move_in_signal_direction": "Avg 5-day move"})
        st.dataframe(sc, width="stretch", column_config={"Right after 5 days": st.column_config.NumberColumn(format="percent"),
                                                         "Avg 5-day move": st.column_config.NumberColumn(format="percent")})
    with st.expander(f"Live data sources ({sum(d['ok'] for d in data_status)} of {len(data_status)} connected)"):
        st.dataframe(pd.DataFrame([{"": "✅" if d["ok"] else "⚠️", "Source": d["name"], "Provides": d["provides"],
                                    "Data as of": d["as_of"] or "—", "Status": d["message"]} for d in data_status]),
                     hide_index=True, width="stretch")

# --- one-line status banner (written last so it can use every tab's results) -----------------------
changes = sum(1 for r in recs if r.action != "HOLD")
top = max(report.trigger_probabilities, key=lambda t: t.p_breach, default=None)
flag_count = sum(1 for stages in flows.values() for sr in stages for i in sr.items if i.level >= WATCH)
risk_txt = (f"highest weather risk: {top.trigger.lower()} {top.p_breach:.0%}" if top and top.p_breach >= 0.10
            else "no weather disruption above 10% in the next 8 days")
banner.markdown(f"{'🔴' if report.alerts else '🟡' if flag_count else '🟢'} **{scenario}** · {risk_txt} · "
                f"{flag_count} supply-chain flag{'s' if flag_count != 1 else ''} · AI suggests {changes} change{'s' if changes != 1 else ''} · "
                f"{len(report.alerts) or 'no'} hedge alert{'s' if len(report.alerts) != 1 else ''}")
