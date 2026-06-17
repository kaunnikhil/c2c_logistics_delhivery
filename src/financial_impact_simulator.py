"""
Financial Risk & Churn Impact Simulator
This module quantifies the financial damage of operational SLA breaches 
by calculating the lost Customer Lifetime Value (LTV) when delays cause churn.
here we focus on the customers who faced the SLA breaches to ultimately reflect the hazard rate of those customers in the survival model
"""

import logging
import pandas as pd
import numpy as np

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FinancialImpactSimulator:
    def __init__(self, ltv_data_path: str, ops_data_path: str):
        self.ltv_data_path = ltv_data_path
        self.ops_data_path = ops_data_path
        
    def calculate_impact(self):
        logger.info("Loading financial and operational datasets")
        
        try:
            ltv_df = pd.read_csv(self.ltv_data_path)
            ops_df = pd.read_csv(self.ops_data_path)
        except FileNotFoundError as e:
            logger.error(f"Missing data file: {e}")
            return
            
        # isolating the C2C Cohort from Operations
        c2c_ops = ops_df[ops_df['Shipment_Type'] == 'C2C_Standard']
        total_c2c_shipments = len(c2c_ops)
        breached_c2c_shipments = c2c_ops['SLA_Breach'].sum()
        
        logger.info(f"Total C2C Shipments Analyzed: {total_c2c_shipments}")
        logger.info(f"C2C Shipments Breaching SLA: {breached_c2c_shipments} ({(breached_c2c_shipments/total_c2c_shipments)*100:.2f}%)")

        # calculate baseline LTV parameters from LTV dataset
        avg_aov = ltv_df['avg_order_value_inr'].mean()
        avg_orders_per_yr = ltv_df['orders_per_year'].mean()
        avg_annual_revenue_per_user = avg_aov * avg_orders_per_yr
        
        # Simulate Churn Impact
        # assumption based on industry standards: an SLA breach increases the likelihood 
        # of a user abandoning the platform by 45% 
        churn_probability_multiplier = 0.45 
        
        # Customers who experienced a delay and subsequently churned
        estimated_churned_users = int(breached_c2c_shipments * churn_probability_multiplier)
        
        # Calculate Future Value Lost (assuming an average lost lifespan of 1.5 years)
        avg_lost_lifespan_years = 1.5
        financial_loss_inr = estimated_churned_users * avg_annual_revenue_per_user * avg_lost_lifespan_years
        
        # Executive Summary
        print("STRATEGIC FINANCIAL IMPACT REPORT: COST OF DELAYS")
        print(f"Network Bottlenecks Resulted in C2C SLA Breaches: {breached_c2c_shipments:,}")
        print(f"Estimated Customers Lost to Competitors (Churn):  {estimated_churned_users:,}")
        print("-" * 60)
        print(f"Average Annual Revenue per User:                  ₹{avg_annual_revenue_per_user:,.2f}")
        print(f"Total Projected Lifetime Value (LTV) Lost:        ₹{financial_loss_inr:,.2f}")
        print("="*60)
        print("CONSULTING RECOMMENDATION:")
        print("Investing in automated dynamic routing to reduce hub congestion by just 5%")
        print(f"could directly recover up to ₹{financial_loss_inr * 0.35:,.2f} in bottom-line revenue.")
        print("="*60 + "\n")


if __name__ == "__main__":
    ltv_path = r"D:\cv_projects_2\c2c_logistics\data\customer_ltv_data.csv"
    ops_path = r"D:\cv_projects_2\c2c_logistics\data\network_operations_data.csv"
    
    simulator = FinancialImpactSimulator(ltv_path, ops_path)
    simulator.calculate_impact()