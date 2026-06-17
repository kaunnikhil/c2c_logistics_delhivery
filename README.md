🚚 **C2C Logistics Strategy & Analytics Engine**

Predicting Chaos Before It Happens (And Saving Millions Doing It)

 Live Links

- Live Executive Dashboard: [https://c2clogisticsdelhivery-lr8sixtklxjbdyzwykfa5d.streamlit.app/]

- GitHub Repository: [https://github.com/kaunnikhil/c2c_logistics_delhivery]

~ **The Problem: The "Shock Absorber" Effect**

B2B logistics is predictable. Consumer-to-Consumer (C2C) logistics is absolute chaos.

When a massive logistics network (like Delhivery or Porter) expands into the C2C market, standard queue logic prioritizes high-paying B2B freight. When hubs hit 95%+ utilization;especially during Monsoon season, C2C parcels act as "shock absorbers" and absorb all the delay.

The Business Impact? Late deliveries cause a 45% spike in the churn hazard rate. Customers leave, and Customer Lifetime Value (LTV) evaporates.

~ **The Solution: An End-to-End Predictive Architecture**

This project isn't just a Jupyter Notebook; it's a 3 part decision engine designed to bridge Operations Research with Financial Forecasting.

1. The Financial Engine (Survival Analysis) 

Tech: lifelines, Kaplan-Meier Fitter, Cox Proportional Hazards.

What it does: Mathematically models the "decay curve" of different customer segments (Homepreneurs vs. Students). Proved that subsidizing an "Escrow" feature significantly reduces the hazard rate of churn.

2. The Operational Engine (XGBoost) 

Tech: xgboost, scikit-learn.

What it does: Simulates 50,000+ shipments across 5 major transit hubs. Trains an XGBClassifier to predict SLA breaches at the exact moment of dispatch.

Note on Data Leakage: Aggressively prevented target leakage by blinding the model to post-dispatch variables (because cheating is for Kaggle, not production).

3. The Executive Command Center (Streamlit) 

Tech: streamlit, plotly.

What it does: An interactive dashboard that allows C-Suite executives to play God. Adjusting the "Hub Congestion" slider instantly recalculates how many SLAs will be saved and translates that directly into recovered ₹ Rupees.

4.(Upcoming) The GenAI Dispatch Copilot 

Tech: Custom Python Agentic Workflow.

What it will do: Automate the morning routine. The AI scans the XGBoost predictions, finds the worst bottleneck (e.g., Mumbai Hub at 98% utilization), and drafts an actionable, human readable briefing for ground managers to reroute trucks before the delay happens.

~ **Strategic Business Insights (The "So What?")**

Data without strategy is just math. Here are the core recommendations derived from this engine:

The Bottleneck Decoupling: XGBoost feature importance proved Hub Utilization outweighs distance. Recommendation: Implement Dynamic SLA Buffering during Monsoons to under-promise and over-deliver.

Defending LTV:CAC: Cox Proportional Hazards proved Escrow users stay longer. Recommendation: Subsidize the first Escrow transaction for Homepreneurs. Absorbing a ₹50 fee is mathematically justified to secure thousands in protected LTV.

The "Delhivery Wale Bhaiya" Strategy: Introduce a 'Premium C2C' routing tier that explicitly bypasses standard logic at hubs >95% utilized, protecting brand trust and capturing price-insensitive demographics.

~ How to Run Locally 

1. Clone the chaos:

git clone [https://github.com/kaunnikhil/c2c_logistics_delhivery.git](https://github.com/kaunnikhil/c2c_logistics_delhivery.git)
cd c2c_logistics_delhivery


2. Install dependencies:

pip install -r requirements.txt


3. Generate Data & Train Models:

python src/data_generator.py
python src/operations_data_generator.py
python src/survival_model.py
python src/xgboost_model.py


4. Launch the Command Center:

streamlit run src/dashboard.py


Built with excessive amounts of coffee and a deep respect for supply chain dynamics.