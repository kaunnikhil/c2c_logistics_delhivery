
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from PIL import Image

# pg configuration
st.set_page_config(page_title="DelhiveryPulse", page_icon="📦", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Force text color inside metric to be dark */
    div[data-testid="metric-container"] > label,
    div[data-testid="metric-container"] > div {
        color: #2C3E50 !important;
    }
    </style>
    """, unsafe_allow_html=True)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")

@st.cache_data
def load_data():
    ops_path = os.path.join(DATA_DIR, 'network_operations_data.csv')
    ltv_path = os.path.join(DATA_DIR, 'customer_ltv_data.csv')
    
    if not os.path.exists(ops_path) or not os.path.exists(ltv_path):
        raise FileNotFoundError(f"Missing data files in {DATA_DIR}. Run generators first.")
        
    ops_df = pd.read_csv(ops_path)
    ltv_df = pd.read_csv(ltv_path)
    
    avg_aov = ltv_df['avg_order_value_inr'].mean()
    avg_orders = ltv_df['orders_per_year'].mean()
    arpu = avg_aov * avg_orders
    return ops_df, ltv_df, arpu

try:
    ops_df, ltv_df, arpu = load_data()
except Exception as e:
    st.error("CRITICAL ERROR: Data Loading Failed")
    st.write(str(e))
    st.stop()

#SIDEBAR
st.sidebar.title("Scenario Planning")
congestion_reduction = st.sidebar.slider(
    "Target Hub Congestion Reduction (%)",
    min_value=0.0, max_value=20.0, value=5.0, step=1.0,
    help="Simulates AI dynamic routing relieving bottlenecked hubs."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Insight:**\n*Reducing congestion non-linearly decreases queue times, rescuing high-margin C2C SLAs.*")

# SIMULATION ENGINE
c2c_ops = ops_df[ops_df['Shipment_Type'] == 'C2C_Standard'].copy()
baseline_breaches = c2c_ops['SLA_Breach'].sum()

prevented_breaches = int(baseline_breaches * (congestion_reduction / 100) * 1.5) 
new_breaches = max(0, baseline_breaches - prevented_breaches)

churn_multiplier = 0.45
avg_lifespan = 1.5

baseline_ltv_lost = int(baseline_breaches * churn_multiplier) * arpu * avg_lifespan
new_ltv_lost = int(new_breaches * churn_multiplier) * arpu * avg_lifespan
revenue_recovered = baseline_ltv_lost - new_ltv_lost

# MAIN LAYOUT
st.title("📦 DelhiveryPulse")

st.markdown("""
### Where Logistics Meets Customer Lifetime Value

Predict SLA breaches before they happen, quantify customer churn risk,
and simulate the financial impact of operational interventions.
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline Revenue at Risk", f"₹{baseline_ltv_lost:,.0f}")
col2.metric("Projected Revenue at Risk", f"₹{new_ltv_lost:,.0f}", f"₹{revenue_recovered:,.0f} recovered", delta_color="inverse")
col3.metric("Current SLA failures", f"{baseline_breaches:,}")
col4.metric("Prevented(simulated) SLA failures", f"{new_breaches:,}", f"-{prevented_breaches:,}", delta_color="inverse")

st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Operational Risk (XGBoost)")
    try:
        st.image(Image.open(os.path.join(VISUALS_DIR, 'xgboost_feature_importance.png')), use_container_width=True)
    except:
        st.warning("XGBoost plot missing. Run `xgboost_model.py` to generate it.")

with col_right:
    st.markdown("### Financial Risk (Survival Analysis)")
    try:
        st.image(Image.open(os.path.join(VISUALS_DIR, 'km_survival_curve.png')), use_container_width=True)
    except:
        st.warning("Survival plot missing. Run `survival_model.py` to generate it.")

st.markdown("---")
st.markdown("### Network Bottleneck Analysis")
hub_breaches = c2c_ops.groupby('Origin_Hub')['SLA_Breach'].mean().reset_index()
hub_breaches['SLA_Breach'] = hub_breaches['SLA_Breach'] * 100

fig = px.bar(
    hub_breaches, 
    x='Origin_Hub', 
    y='SLA_Breach',
    title="C2C SLA Breach Rate by Origin Hub (%)",
    labels={'SLA_Breach': 'Breach Rate (%)', 'Origin_Hub': 'Hub Location'},
    color='SLA_Breach',
    color_continuous_scale='Reds'
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("## Strategic Action Plan (Final Business Insights)")
tab1, tab2, tab3 = st.tabs(["Operational Interventions", "Financial & Product Strategy", "Go-To-Market"])

with tab1:
    st.markdown("#### The Bottleneck Decoupling Strategy\n* **Diagnosis:** XGBoost proves Hub Utilization outweighs absolute distance in causing delays.\n* **Action:** Implement Dynamic SLA Buffering during Monsoon season.\n* **ROI:** Prevents the psychological trigger that Survival analysis shows spikes customer churn by 45%.")
with tab2:
    st.markdown("#### Defending the LTV:CAC Ratio\n* **Diagnosis:** Survival Analysis shows steep drop-offs for non-Escrow users.\n* **Action:** Subsidize the first Escrow transaction for Homepreneurs.\n* **ROI:** Absorbing a ₹50 fee is mathematically justified to secure thousands in protected LTV.")
with tab3:
    st.markdown("#### Expanding Brand Trust\n* **Diagnosis:** Standard C2C parcels absorb B2B queue overflow.\n* **Action:** Introduce 'Premium C2C' tier bypassing standard logic at hubs >95% utilized.\n* **ROI:** Captures price-insensitive demographics and organically relieves network pressure.")