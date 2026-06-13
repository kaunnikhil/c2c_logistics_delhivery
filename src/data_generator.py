import numpy as np
import pandas as pd
import os

def generate_ltv_data(n_customers=10000):
    np.random.seed(42)
    
    # FPDR customer segmentation
    segments = ['Homepreneurs', 'Students', 'Ad-Hoc Sellers']
    customer_segment = np.random.choice(segments, p=[0.3, 0.4, 0.3], size=n_customers)
    
    # acquisition channels and CAC
    channels = ['Organic', 'Paid Social', 'Campus Activation']
    acq_channel = np.random.choice(channels, p=[0.2, 0.6, 0.2], size=n_customers)
    cac_map = {'Organic': 0, 'Paid Social': 150, 'Campus Activation': 50}
    acquisition_cost = [cac_map[c] for c in acq_channel]
    
    # escrow adoption
    uses_escrow = np.random.choice([1, 0], p=[0.4, 0.6], size=n_customers)
    
    tenure_days = np.zeros(n_customers)
    churn_event = np.zeros(n_customers)
    
    for i in range(n_customers):
        # base scale adjusts survival probability based on segment to mimic the frequency and the predictability matrix
        if customer_segment[i] == 'Homepreneurs':
            base_scale = 500 
        elif customer_segment[i] == 'Students':
            base_scale = 180 
        else:
            base_scale = 45 
            
        if uses_escrow[i] == 1:
            base_scale *= 1.4 # escrow increases retention
        #drawn_tenure is a 'time to event' data
        drawn_tenure = np.random.exponential(scale=base_scale)
        observation_window = 365 # 1 year timeline
        
        if drawn_tenure > observation_window:
            tenure_days[i] = observation_window
            churn_event[i] = 0 # still active
        else:
            tenure_days[i] = drawn_tenure
            churn_event[i] = 1 # churned
            
    # financials (aov and frequency)
    base_aov = np.random.normal(120, 20, size=n_customers)
    aov = base_aov + (uses_escrow * 25) 
    order_freq_per_year = np.zeros(n_customers)
    for i in range(n_customers):
        if customer_segment[i] == 'Homepreneurs':
            # 24 orders/year, varies naturally per user via poisson
            order_freq_per_year[i] = np.random.poisson(lam=24) 
        elif customer_segment[i] == 'Students':
            # same as homepreneurs, but comparitively lesser lambda to decrease the average number of order frequency
            order_freq_per_year[i] = np.random.poisson(lam=4)
        else:
            order_freq_per_year[i] = np.random.poisson(lam=2)
    
    df = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'segment': customer_segment,
        'acquisition_channel': acq_channel,
        'uses_escrow': uses_escrow,
        'acquisition_cost_inr': acquisition_cost,
        'avg_order_value_inr': np.round(aov, 2),
        'orders_per_year': order_freq_per_year,
        'tenure_days': np.round(tenure_days, 0),
        'churned': churn_event.astype(int)
    })
    
    output_path = r"D:\cv_projects_2\c2c_logistics\data\customer_ltv_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    generate_ltv_data()