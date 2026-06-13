import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import os

def run_survival_analysis():
    data_path = r"D:\cv_projects_2\c2c_logistics\data\customer_ltv_data.csv"
    visuals_dir = r"D:\cv_projects_2\c2c_logistics\visuals"
    os.makedirs(visuals_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Run data_generator.py first.")
        return
    # Kaplan-Meier Analysis (Visualizing Cohort Churn)

    print("Fitting Kaplan-Meier Survival Curves...")
    kmf = KaplanMeierFitter()
    
    plt.figure(figsize=(10, 6))
    
    # Plot a survival curve for each segment
    for segment in df['segment'].unique():
        mask = df['segment'] == segment
        kmf.fit(df['tenure_days'][mask], event_observed=df['churned'][mask], label=segment)
        kmf.plot_survival_function(ci_show=True) # ci_show adds confidence intervals

    plt.title('C2C Logistics: Customer Retention over Time by Segment', fontsize=14, fontweight='bold')
    plt.xlabel('Days since First Shipment', fontsize=12)
    plt.ylabel('Probability of Remaining Active', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Save the plot
    km_plot_path = os.path.join(visuals_dir, 'km_survival_curve.png')
    plt.tight_layout()
    plt.savefig(km_plot_path, dpi=300)
    plt.close()
    print(f"Kaplan-Meier plot saved to: {km_plot_path}")

    # ---------------------------------------------------------
    # 3. Cox Proportional Hazards (Feature Impact)
    # ---------------------------------------------------------
    print("\nFitting Cox Proportional Hazards Model...")
    
    # Prepare data: Cox models require categorical variables to be one-hot encoded (dummies)
    # We drop 'customer_id' and financial metrics since we are only predicting tenure based on behavior/acquisition
    cox_df = df[['segment', 'acquisition_channel', 'uses_escrow', 'tenure_days', 'churned']]
    cox_df = pd.get_dummies(cox_df, columns=['segment', 'acquisition_channel'], drop_first=True)
    
    # Initialize and fit the model
    cph = CoxPHFitter(penalizer=0.1) # Added a small penalizer for numerical stability
    cph.fit(cox_df, duration_col='tenure_days', event_col='churned')

    # Print the statistical summary to the terminal
    print("\n--- Cox Proportional Hazards Model Summary ---")
    cph.print_summary()

    # Plot the Hazard Ratios
    plt.figure(figsize=(10, 6))
    cph.plot()
    plt.title('Impact of Features on Churn Risk (Hazard Ratios)', fontsize=14, fontweight='bold')
    plt.axvline(0, color='red', linestyle='--', alpha=0.5) # Add a baseline reference
    
    # Save the plot
    cox_plot_path = os.path.join(visuals_dir, 'cox_hazard_ratios.png')
    plt.tight_layout()
    plt.savefig(cox_plot_path, dpi=300)
    plt.close()
    print(f"Cox Hazard Ratios plot saved to: {cox_plot_path}")

if __name__ == "__main__":
    run_survival_analysis()