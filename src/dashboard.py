import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from PIL import Image

# pg config
st.set_page_config(page_title="C2C Logistics Command Center", page_icon="📦", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .stMetric div {color: #2C3E50 !important;}
    </style>
    """, unsafe_allow_html=True)

# dynamic paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
VISUALS_DIR = os.path.join(BASE_DIR, 'visuals')

@st.cache_data
def load_data():
    ops_df = pd.read_csv(os.path.join(DATA_DIR, 'network_operations_data.csv'))
    ltv_df = pd.read_csv(os.path.join(DATA_DIR, 'customer_ltv_data.csv'))
    
    avg_aov = ltv_df['avg_order_value_inr'].mean()
    avg_orders = ltv_df['orders_per_year'].mean()
    arpu = avg_aov * avg_orders
    return ops_df, ltv_df, arpu

try:
    ops_df, ltv_df, arpu = load_data()
except Exception as e:
    st.error(f"CRITICAL ERROR: Data Loading Failed\n\nMissing data files in {DATA_DIR}. Error: {e}")
    st.stop()

# sidebar
st.sidebar.title(" Scenario Planning")
congestion_reduction = st.sidebar.slider(
    "Target Hub Congestion Reduction (%)",
    min_value=0.0, max_value=20.0, value=5.0, step=1.0,
    help="Simulates AI dynamic routing relieving bottlenecked hubs."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Consulting Insight:**\n*Reducing congestion non-linearly decreases queue times, rescuing high-margin C2C SLAs.*")

#simulation
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
st.title("C2C Logistics Expansion: Strategic Command Center")

st.markdown("### Financial Impact of Operational Delays")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline LTV Risk", f"₹{baseline_ltv_lost:,.0f}")
col2.metric("Simulated LTV Risk", f"₹{new_ltv_lost:,.0f}", f"₹{revenue_recovered:,.0f} recovered", delta_color="inverse")
col3.metric("Baseline C2C Breaches", f"{baseline_breaches:,}")
col4.metric("Simulated C2C Breaches", f"{new_breaches:,}", f"-{prevented_breaches:,}", delta_color="inverse")

st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Operational Risk (XGBoost)")
    try:
        st.image(Image.open(os.path.join(VISUALS_DIR, 'xgboost_feature_importance.png')), use_container_width=True)
    except:
        st.warning("XGBoost plot missing.")

with col_right:
    st.markdown("### Financial Risk (Survival Analysis)")
    try:
        st.image(Image.open(os.path.join(VISUALS_DIR, 'km_survival_curve.png')), use_container_width=True)
    except:
        st.warning("Survival plot missing.")

st.markdown("---")
st.markdown("##  Strategic Action Plan (Consulting Insights)")
tab1, tab2, tab3 = st.tabs([" Operational Interventions", " Financial & Product Strategy", " Go-To-Market"])

with tab1:
    st.markdown("#### The Bottleneck Decoupling Strategy\n* **Diagnosis:** XGBoost proves `Hub_Utilization` and `Season_Monsoon` overshadow absolute distance in causing delays. C2C parcels are currently acting as a buffer for B2B freight during peak congestion.\n* **Action:** Implement **Dynamic SLA Buffering**. During Monsoon season, automatically inject a 12-hour buffer into the promised delivery date shown to C2C customers originating from high-risk hubs.\n* **ROI:** Under-promising and over-delivering prevents the 'SLA Breach' psychological trigger that our Survival model shows spikes customer churn by 45%.")
with tab2:
    st.markdown("#### Defending the LTV:CAC Ratio\n* **Diagnosis:** Survival Analysis indicates a steep drop-off for 'Students' and 'Ad-Hoc' segments, whereas 'Homepreneurs' exhibit a much flatter decay curve if they use Escrow.\n* **Action:** **Gamify Escrow Adoption.** Shift marketing spend away from generic acquisition and heavily subsidize the first Escrow transaction for Homepreneurs.\n* **ROI:** Absorbing a ₹50 transaction fee is mathematically justified to secure thousands in protected LTV by lowering the churn hazard rate.")
with tab3:
    st.markdown("#### Expanding the 'Delhivery Wale Bhaiya' Brand\n* **Diagnosis:** C2C logistics relies entirely on trust. A damaged or late package in B2B is an SLA penalty; in C2C, it is permanent brand damage.\n* **Action:** **Tiered C2C Routing.** Introduce a 'Premium C2C' tier that explicitly bypasses standard queue logic at hubs >95% utilized, maintaining the 'Delhivery Wale Bhaiya' promise of reliability.\n* **ROI:** Captures price-insensitive demographics (e.g., sending urgent documents or gifts), increasing Average Order Value (AOV) while organically relieving pressure on primary transit arteries.")