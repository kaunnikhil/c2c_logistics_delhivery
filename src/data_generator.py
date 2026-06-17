import numpy as np
import pandas as pd
import os

def generate_ltv_data(n_customers=10000):
 
    rng = np.random.default_rng(1)
    
    segments = ['Homepreneurs', 'Students', 'Ad-Hoc Sellers']
    customer_segment = rng.choice(segments, p=[0.3, 0.4, 0.3], size=n_customers)
    
    channels = ['Organic', 'Paid Social', 'Campus Activation']
    acq_channel = rng.choice(channels, p=[0.2, 0.6, 0.2], size=n_customers)
    
    cac_map = {'Organic': 0, 'Paid Social': 150, 'Campus Activation': 50}
    acquisition_cost = np.vectorize(cac_map.get)(acq_channel)
    
    # escrow usage will be comparitively lesser and hence divided in 40:60 ratio
    uses_escrow = rng.choice([1, 0], p=[0.4, 0.6], size=n_customers)
    
    # vectorized Tenure and Churn Generation
    # Map segments to base scales
    base_scale = np.select(
        [customer_segment == 'Homepreneurs', customer_segment == 'Students'], [500, 90],
        default=45
    )
    
    # escrow multiplier 
    final_scale = np.where(uses_escrow == 1, base_scale * 1.4, base_scale)
    
    # tenure taken from random exponential
    drawn_tenure = rng.exponential(scale=final_scale)
    observation_window = 365
    
    # capped tenure at 365 days and flag active users
    tenure_days = np.minimum(drawn_tenure, observation_window)
    churn_event = np.where(drawn_tenure > observation_window, 0, 1)
            
    # aov and order Frequency
    base_aov = rng.normal(120, 20, size=n_customers)
    aov = base_aov + (uses_escrow * 25) 
    

    # Homepreneurs: shape=6, scale = 4  avg 24 orders a yr
    # Students: shape=4, scale=1  avg 4 orders a yr
    # Ad-Hoc Sellers: shape=2, scale=1  avg 2 orders a yr
    gamma_shapes = np.select([customer_segment == 'Homepreneurs', customer_segment == 'Students'], [6, 4], default=2)
    gamma_scales = np.select([customer_segment == 'Homepreneurs', customer_segment == 'Students'], [4, 1], default=1)
    
    #generate individual user lambdas (Negative Binomial equivalent)
    user_lambdas = rng.gamma(shape=gamma_shapes, scale=gamma_scales)
    
    # vectorized poisson generation utilizing the dynamic lambdas
    order_freq_per_year = rng.poisson(lam=user_lambdas)
    
    # 5. Build and Export DataFrame
    df = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'segment': customer_segment,
        'acquisition_channel': acq_channel,
        'uses_escrow': uses_escrow,
        'acquisition_cost_inr': acquisition_cost,
        'avg_order_value_inr': np.round(aov, 2),
        'orders_per_year': order_freq_per_year,
        'tenure_days': np.round(tenure_days, 0).astype(int),
        'churned': churn_event
    })
    
    output_path = r"D:\cv_projects_2\c2c_logistics\data\customer_ltv_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved {n_customers} rows to {output_path}")

if __name__ == "__main__":
    generate_ltv_data()
