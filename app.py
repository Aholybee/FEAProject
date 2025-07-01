import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import io

# Page config
st.set_page_config(page_title="Executive Financial Strategy", layout="wide")

# Header styling
st.markdown("""
<style>
.big-header {
    font-size:2.3em;
    font-weight:700;
    color:#003366;
    margin-bottom:0;
}
.sub-header {
    font-size:1.2em;
    font-weight:400;
    color:#666666;
    margin-top:0;
}
hr {margin-top:0.5rem; margin-bottom:1.5rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-header">📊 Executive Financial Strategy Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dynamic decision-making platform with correlation-aware risk modeling and probabilistic forecasting</div>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Controls")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Years)", 0.25, 5.0, 1.0)
steps = st.sidebar.slider("Trading Days", 50, 252, 252)
n_sim = st.sidebar.slider("Simulations", 10, 500, 50)
model_choice = st.sidebar.radio("Sharpe Ratio Model", ["Naïve (No Correlation)", "Correlation-Aware"])

st.sidebar.header("📤 Upload Asset Template")

download_link = "sample_data/Financial_Template.xlsx"
with open(download_link, "rb") as file:
    btn = st.sidebar.download_button(
        label="📥 Download Excel Template",
        data=file,
        file_name="Financial_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

uploaded_file = st.sidebar.file_uploader("📂 Upload your filled template", type=["xlsx"])

st.sidebar.header("📤 Template Upload & Download")

# Download button
with open("sample_data/Financial_Dashboard_Template.xlsx", "rb") as f:
    st.sidebar.download_button(
        label="📥 Download Dashboard Template",
        data=f,
        file_name="Financial_Dashboard_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# File uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload your completed template", type=["xlsx"])

if uploaded_file:
    try:
        xl = pd.ExcelFile(uploaded_file)
        asset_df = xl.parse("Portfolio Assets")
        income_df = xl.parse("Income Statement")
        st.sidebar.success("✅ Template uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error parsing uploaded file: {e}")
        st.stop()
else:
    # Default fallback for simulation
    xl = pd.ExcelFile("sample_data/Financial_Dashboard_Template.xlsx")
    asset_df = xl.parse("Portfolio Assets")
    income_df = xl.parse("Income Statement")

# Load asset inputs from user file
if uploaded_file:
    try:
        xl = pd.ExcelFile(uploaded_file)
        asset_df = xl.parse("Portfolio Assets")
        st.sidebar.success("✅ Template uploaded successfully!")
    except Exception as e:
        st.sidebar.error("⚠️ Failed to read 'Portfolio Assets' sheet. Please check formatting.")
        st.stop()
else:
    asset_df = pd.read_excel(download_link, sheet_name="Portfolio Assets")




# Simulated asset inputs (can be replaced with uploads)
asset_inputs = [
    {"Asset": "Asset A", "Expected Return (%)": 8, "Volatility (%)": 20, "Value": 100, "Custom Weight (%)": 25},
    {"Asset": "Asset B", "Expected Return (%)": 6, "Volatility (%)": 15, "Value": 100, "Custom Weight (%)": 25},
    {"Asset": "Asset C", "Expected Return (%)": 10, "Volatility (%)": 25, "Value": 100, "Custom Weight (%)": 25},
    {"Asset": "Asset D", "Expected Return (%)": 7, "Volatility (%)": 18, "Value": 100, "Custom Weight (%)": 25},
]

asset_df = pd.DataFrame(asset_inputs)

def style_sharpe(ratio):
    if np.isnan(ratio): return "⚠️ N/A"
    elif ratio > 1.0: return f"🟢 {ratio:.2f}"
    elif ratio > 0.0: return f"🟡 {ratio:.2f}"
    else: return f"🔴 {ratio:.2f}"

def calculate_sharpe(data, risk_free, cov=None):
    r = np.array([a["Expected Return (%)"] for a in data])
    v = np.array([a["Volatility (%)"] for a in data])
    w = np.array([a["Custom Weight (%)"] for a in data])
    w = w / w.sum()
    pr = np.dot(w, r)
    if cov is not None:
        w = w.reshape(-1,1)
        pv = np.sqrt((w.T @ cov @ w)[0][0])
    else:
        pv = np.sqrt(np.dot(w**2, v**2))
    return round((pr - risk_free) / pv, 4) if pv > 0 else np.nan

def simulate_gbm(S0, mu, sigma, T, steps, n_sim):
    dt = T / steps
    Z = np.random.standard_normal((steps, n_sim))
    S = np.zeros_like(Z)
    S[0] = S0
    for t in range(1, steps):
        S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t])
    return pd.DataFrame(S)

def simulate_portfolio(asset_inputs, steps, n_sim, T):
    sim = {}
    for a in asset_inputs:
        mu = a["Expected Return (%)"] / 100
        sigma = a["Volatility (%)"] / 100
        sim[a["Asset"]] = simulate_gbm(a["Value"], mu, sigma, T, steps, n_sim)
    return sim

def combine_portfolio(sim, weights):
    keys = list(sim.keys())
    n_steps, n_sim = sim[keys[0]].shape
    total = np.zeros((n_steps, n_sim))
    for i, k in enumerate(keys):
        total += sim[k].values * weights[i]
    return pd.DataFrame(total)

def flag_corr(corr, threshold=0.85):
    flagged = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                flagged.append((corr.columns[i], corr.columns[j], val))
    return flagged

def simulate_scenarios(inputs, steps, n_sim, T, scenarios):
    results = {}
    for label, adj in scenarios.items():
        adjusted = []
        for a in inputs:
            adjusted.append({
                **a,
                "Expected Return (%)": a["Expected Return (%)"] * adj["return_adj"],
                "Volatility (%)": a["Volatility (%)"] * adj["vol_adj"]
            })
        sim = simulate_portfolio(adjusted, steps, n_sim, T)
        w = np.array([a["Custom Weight (%)"] for a in adjusted])
        w = w / w.sum()
        df = combine_portfolio(sim, w)
        results[label] = df.mean(axis=1)
    return pd.DataFrame(results)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Simulations", "🔮 Scenarios"])

with tab1:
    st.subheader("📌 Asset Overview")
    st.dataframe(asset_df)

    returns_df = pd.DataFrame({
        a["Asset"]: np.random.normal(a["Expected Return (%)"]/100/252, a["Volatility (%)"]/100/np.sqrt(252), size=252)
        for a in asset_inputs
    })

    cov = returns_df.cov()
    corr = returns_df.corr()
    weights = np.array([a["Custom Weight (%)"] for a in asset_inputs])
    weights = weights / weights.sum()

    sharpe_val = calculate_sharpe(asset_inputs, risk_free_rate, cov if model_choice == "Correlation-Aware" else None)
    st.markdown(f"**📈 Portfolio Sharpe Ratio:** {style_sharpe(sharpe_val)}")

    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    st.pyplot(fig)

    st.subheader("📋 Diversification Insights")
    issues = flag_corr(corr)
    if issues:
        for a1, a2, val in issues:
            st.warning(f"{a1} and {a2} show high correlation ({val:.2f})")
    else:
        st.success("✅ No correlation issues detected")

with tab2:
    st.subheader("📊 Monte Carlo Portfolio Simulation")
    simulations = simulate_portfolio(asset_inputs, steps, n_sim, forecast_horizon)
    port_df = combine_portfolio(simulations, weights)
    st.line_chart(port_df)

    st.subheader("🪞 Asset Comparison")
    selected = st.multiselect("Choose Assets", list(simulations.keys()), default=list(simulations.keys())[:2])
    fig = go.Figure()
    for a in selected:
        fig.add_trace(go.Scatter(y=simulations[a].mean(axis=1), mode='lines', name=a))
    fig.update_layout(title="Average Simulated Paths", xaxis_title="Time", yaxis_title="Price ($)")
    st.plotly_chart(fig)

with tab3:
    st.header("🔮 Scenario Comparison")
    scenarios = {
        "Optimistic": {"return_adj": 1.25, "vol_adj": 0.9},
        "Base Case": {"return_adj": 1.0, "vol_adj": 1.0},
        "Pessimistic": {"return_adj": 0.75, "vol_adj": 1.2}
    }
    scenario_df = simulate_scenarios(asset_inputs, steps, n_sim, forecast_horizon, scenarios)
    fig = px.line(scenario_df, labels={"value": "Portfolio Value", "index": "Time (Days)"})
    fig.update_layout(title="Forecast Scenarios", legend_title_text="Scenario")
    st.plotly_chart(fig)

    st.subheader("📈 Insight Summary")
    final_vals = scenario_df.iloc[-1]
    base = final_vals["Base Case"]
    for s in scenario_df.columns:
        if s == "Base Case":
            st.markdown(f"🟦 **{s}**: Final Value = ${base:,.2f}")
        else:
            delta = final_vals[s] - base
            sign = "📈" if delta > 0 else "📉"
            color = "🟢" if delta > 0 else "🔴"
            st.markdown(f"{color} **{s}**: ${final_vals[s]:,.2f} ({sign} {delta:+,.2f} vs Base Case)")

    # Optional commentary
    if final_vals["Optimistic"] > 1.2 * base:
        st.success("🚀 Optimistic case shows strong upside potential (+20% over base)")
    if final_vals["Pessimistic"] < 0.9 * base:
        st.warning("⚠️ Pessimistic case indicates downside risk—consider stress testing defensive allocations.")