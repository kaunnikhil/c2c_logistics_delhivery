📦 **C2C Logistics Expansion: Strategic Analytics Engine**

Live Executive Dashboard: [https://c2clogisticsdelhivery-lr8sixtklxjbdyzwykfa5d.streamlit.app/]
📌 Executive Summary

This project simulates the strategic and operational challenges of a major B2B logistics provider (e.g., Delhivery, Porter, Blue Dart) expanding into the high-margin Consumer-to-Consumer (C2C) market.

By bridging Operations Research (Machine Learning) with Financial Forecasting (Survival Analysis), this pipeline quantifies exactly how physical network bottlenecks destroy Customer Lifetime Value (LTV) and provides actionable, data-backed consulting recommendations to recover revenue.

~ **Architecture & Methodologies**

1. Financial Risk Engine (Survival Analysis)

Objective: Quantify the exact cost of customer churn.

Tech: lifelines (Kaplan-Meier Fitter & Cox Proportional Hazards).

Impact: Modeled segment-specific decay curves (Homepreneurs vs. Students) and proved mathematically that adopting an 'Escrow' feature drastically reduces the hazard rate of customer churn.

2. Operational ML Engine (XGBoost)

Objective: Predict SLA breaches before the truck leaves the origin hub.

Tech: XGBClassifier with dynamic scale_pos_weight for extreme class imbalance.

Data Constraints (Zero Leakage): Aggressively prevented target leakage by training the model only on variables known at $t=0$ (Origin Hub Congestion, Weather Seasonality, Route Risk).

Impact: Extracted Gain-based feature importances to show ground operations exactly why a delay was flagged.

3. Strategic Command Center (Streamlit)

Objective: An interactive executive dashboard linking ML predictions to financial ROI.

Impact: Proves to stakeholders that a 5% reduction in origin hub congestion via dynamic routing doesn't just improve an engineering metric—it directly recovers ₹1.95 Million+ in bottom-line LTV.

~ How to Run Locally

Clone the repository:

git clone https://github.com/kaunnikhil/c2c_logistics_delhivery.git
cd c2c_logistics


Install dependencies:

pip install -r requirements.txt


Launch the Executive Dashboard:

streamlit run src/dashboard.py


~ **Strategic Business Outcomes**

The Bottleneck Decoupling Strategy: XGBoost feature importance proves Hub_Utilization overshadows distance in causing delays. Standard C2C parcels currently absorb B2B queue overflow.

LTV Defense: Financial simulation proves that subsidizing the first Escrow transaction for Homepreneurs mathematically defends the LTV:CAC ratio.