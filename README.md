# C2C Logistics Strategy & Analytics Engine

### From Market Entry Strategy to Predictive Decision Support

This repository extends our solution for **The Delhivery Way**, a national case competition conducted by E-Cell IIT Kharagpur, into an end-to-end business analytics platform.

The original challenge focused on designing a scalable strategy for Delhivery's expansion into the Consumer-to-Consumer (C2C) logistics market while preserving service quality, customer trust, and unit economics. Rather than stopping at strategic recommendations, this project translates those recommendations into quantitative models capable of simulating operational outcomes, customer behaviour, and financial impact.

The repository combines strategic consulting, financial modelling, machine learning, survival analysis, and interactive visualization into a single decision-support system.

---

## Live Demo

**Executive Dashboard**

https://c2clogisticsdelhivery-lr8sixtklxjbdyzwykfa5d.streamlit.app/

**GitHub Repository**

https://github.com/kaunnikhil/c2c_logistics_delhivery

---

# Business Context

Delhivery's logistics network has historically been optimized for B2B operations. Entering the C2C logistics market introduces a fundamentally different operational challenge:

- Highly unpredictable shipment patterns
- Lower shipment density
- Greater variability in pickup locations
- Higher customer service expectations
- Increased sensitivity to delivery delays

As network utilization increases, C2C shipments become disproportionately affected by congestion, leading to missed SLAs, declining customer trust, and ultimately customer churn.

This project investigates how operational decisions influence customer retention and profitability while providing executives with data-driven recommendations for managing these trade-offs.

---

# Project Objectives

The analytics engine is designed to answer six core business questions:

- Which customer segments should Delhivery prioritize during market entry?
- Which operational variables contribute most to SLA breaches?
- How does hub congestion affect delivery performance?
- What is the financial impact of delayed deliveries?
- Which interventions generate the highest long-term customer value?
- How can executives proactively prevent network bottlenecks?

---

# Repository Structure

```
.
├── data/
│   ├── customer_ltv_data/
│   └── network_operations_data/
│
├── THE DELHIVERY WAY/
│   ├── THE DELHIVERY WAY - Financial Model.pdf
│   └── THE DELHIVERY WAY - Final Submission.xlsx
│
├── src/
│   ├── data_generator
│   ├── dashboard
│   ├── survival_model
│   ├── xgb_model
│   ├── financial_impact_simulator
│   └── operations_data_generator
│
├── visuals/
│
├── README.md
└── requirements.txt
```

---

# Project Architecture

```
Delhivery Case Study
          │
          ▼
 Market Research & Strategy
          │
          ▼
 Financial Model
          │
          ▼
 Synthetic Operations Data
          │
          ▼
 Predictive ML Models
          │
          ▼
 Executive Dashboard
          │
          ▼
 Strategic Recommendations
```

---

# Components

## 1. Strategy Consulting

The repository includes the complete consulting solution prepared during the competition covering:

- Market sizing
- Competitive benchmarking
- Customer segmentation (FPDR Framework)
- Go-to-market strategy
- Service differentiation
- Consumer journey design
- Strategic partnerships
- Marketing strategy
- National rollout roadmap

These recommendations serve as the business foundation for the analytics models developed in this repository.

---

## 2. Financial Model

A detailed financial model was developed to validate the commercial feasibility of the proposed strategy.

The model evaluates:

- Customer acquisition economics
- Unit economics
- Contribution margins
- Customer Lifetime Value (LTV)
- Cost assumptions
- Operational profitability
- Business scalability under different growth scenarios

Instead of treating analytics independently, the predictive models are evaluated against measurable business outcomes.

---

## 3. Customer Retention Engine

**Technology**

- lifelines
- Kaplan-Meier Survival Analysis
- Cox Proportional Hazards Model

The survival analysis estimates how customer retention changes across different customer segments and operational experiences.

Rather than measuring only immediate churn, it quantifies long-term customer value and evaluates the effectiveness of strategic interventions on customer lifetime.

---

## 4. Operational Risk Engine

**Technology**

- Python
- XGBoost
- scikit-learn

A synthetic logistics network consisting of 50,000+ shipments across multiple hubs was created to simulate C2C operations.

The classification model predicts potential SLA breaches before dispatch using only operational variables available at shipment creation.

Special attention was given to preventing target leakage by excluding all post-dispatch information during model training.

---

## 5. Executive Command Center

**Technology**

- Streamlit

An interactive dashboard enables business users to evaluate operational scenarios in real time.

---

## 6. Upcoming Module — AI Operations Copilot

An agentic workflow is currently under development to automate operational decision support.

The planned system will:

- Analyze predicted SLA breaches
- Identify high-risk logistics hubs
- Prioritize operational interventions
- Generate executive briefings
- Recommend proactive rerouting strategies

---

# Key Business Insights

The analytics pipeline produced several actionable recommendations.

### Dynamic SLA Buffering

Hub utilization contributes more to delivery delays than shipment distance.

Introducing dynamic SLA commitments during periods of high congestion can improve customer experience while reducing operational pressure.

---

### Customer Retention Strategy

Survival analysis indicates that customer trust mechanisms significantly improve long-term retention.

Investment in trust-building features produces stronger long-term returns than purely acquisition-focused incentives.

---

### Operational Prioritization

Not every shipment contributes equally to future profitability.

Prioritizing high-value customer segments during periods of constrained network capacity maximizes long-term business value.

---

# Technology Stack

**Programming**

- Python

**Machine Learning**

- XGBoost
- scikit-learn

**Statistical Modeling**

- lifelines
- Kaplan-Meier
- Cox Proportional Hazards

**Data Analysis**

- pandas
- NumPy

**Visualization**

- Plotly
- Streamlit

---

# Supporting Documents

This repository also includes the original business artefacts developed during the competition.

| Document | Description |
|-----------|-------------|
| Delhivery Case Statement | Original competition problem statement |
| Strategy Solution | Complete consulting solution including market analysis, customer segmentation, execution roadmap and GTM strategy |
| Financial Model | Excel model validating unit economics, pricing assumptions and business scalability |

Together, these documents provide the complete journey from business problem identification to analytical implementation.

---

# Running the Project

Clone the repository

```bash
git clone https://github.com/kaunnikhil/c2c_logistics_delhivery.git
cd c2c_logistics_delhivery
```

Install dependencies

```bash
pip install -r requirements.txt
```

Generate synthetic datasets

```bash
python src/data_generator.py
python src/operations_data_generator.py
```

Train models

```bash
python src/survival_model.py
python src/xgboost_model.py
```

Launch the dashboard

```bash
streamlit run src/dashboard.py
```

---

# Future Work

- Agentic AI Operations Copilot
- Real-time streaming data pipeline
- Digital twin of logistics network
- Network optimization using Operations Research
- Scenario planning under stochastic demand
- Reinforcement Learning for routing decisions

---

# Disclaimer

This project was developed for educational and research purposes as an extension of a national case competition. The operational datasets used in the machine learning pipeline are synthetic and were generated to simulate realistic logistics scenarios. They do not represent proprietary operational data from Delhivery.