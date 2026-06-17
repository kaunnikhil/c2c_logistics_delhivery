"""
this script generates synthetic shipment histories, transit metrics, and SLA breach statuses
across 5 hubs (Delhi, Mumbai, Bengaluru, Kolkata, Hyderabad) incorporating
compounding queue delays, weather shocks, and priority tier structures.

"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict

#configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OperationsDataGenerator:
    """
    Generator class to simulate operations, congestion, weather risks, and SLA breaches
    for the logistics network.
    """

    # Static hub names
    HUBS = ['Delhi_Hub', 'Mumbai_Hub', 'Bengaluru_Hub', 'Kolkata_Hub', 'Hyderabad_Hub']

    # static city-pair metrics (distance, base terrain/weather risk , traffic unpredictability, road reliability, terrain difficulty)
    # keys are alphabetically sorted tuples to ensure symmetry
    ROUTE_METRICS = {
        ('Bengaluru_Hub', 'Delhi_Hub'): {'distance': 2100.0, 'base_risk': 0.5},
        ('Bengaluru_Hub', 'Hyderabad_Hub'): {'distance': 600.0, 'base_risk': 0.3},
        ('Bengaluru_Hub', 'Kolkata_Hub'): {'distance': 1800.0, 'base_risk': 0.6},
        ('Bengaluru_Hub', 'Mumbai_Hub'): {'distance': 1000.0, 'base_risk': 0.5},
        ('Delhi_Hub', 'Hyderabad_Hub'): {'distance': 1500.0, 'base_risk': 0.4},
        ('Delhi_Hub', 'Kolkata_Hub'): {'distance': 1500.0, 'base_risk': 0.7},
        ('Delhi_Hub', 'Mumbai_Hub'): {'distance': 1400.0, 'base_risk': 0.6},
        ('Hyderabad_Hub', 'Kolkata_Hub'): {'distance': 1500.0, 'base_risk': 0.5},
        ('Hyderabad_Hub', 'Mumbai_Hub'): {'distance': 700.0, 'base_risk': 0.4},
        ('Kolkata_Hub', 'Mumbai_Hub'): {'distance': 1900.0, 'base_risk': 0.8},
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed=seed)
        logger.info(f"initialized OperationsDataGenerator with seed {seed}")

    def _get_route_properties(self, origin: str, dest: str) -> Tuple[float, float]:
        """
        Lookup distance and base risk for a given origin-destination pair.
        """
        if origin == dest:
            raise ValueError(f"Origin and Destination hubs cannot be the same: {origin}")
        
        pair = (origin, dest) if origin < dest else (dest, origin)
        metrics = self.ROUTE_METRICS.get(pair)
        if metrics is None:
            raise KeyError(f"Route properties not found for pair: {pair}")
        
        return metrics['distance'], metrics['base_risk']

    def compute_congestion_delay(self, utilization: np.ndarray, shipment_type: np.ndarray) -> np.ndarray:
        """
        Compute queue delay penalty at a hub based on utilization and shipment priority tier.

        Rules:
        - B2B shipments bypass minor queues (utilization < 1.05) and receive a 0.25x penalty
          under severe congestion (utilization >= 1.05).
        - C2C shipments face a 1.2x penalty under heavy congestion (utilization > 0.95).
        i.e. full explanation - 
        1. it utilization is under 95 percent then no delay
        2. if it is greater than 95 percent - (calculated base delay)
                if B2B - and utilization > 105 percent then delay = 25 percent of base delay
                if B2B - and utilization < 105 percent then delay = 0 (priority lanes)
                if c2c - delay = 1.25 times the base delay 
        """
        delays = np.zeros(len(utilization))
        congestion_mask = utilization > 0.95
        
        if np.any(congestion_mask):
            # base exponential delay penalty
            base_delays = 4.0 * np.exp(5.0 * (utilization[congestion_mask] - 0.95))
            
            is_b2b = (shipment_type[congestion_mask] == 'B2B_Priority')
            utils_cong = utilization[congestion_mask]
            
            sub_delays = np.zeros(len(base_delays))
            
            # C2C Standard: takes full 1.2x penalty
            sub_delays[~is_b2b] = 1.2 * base_delays[~is_b2b]
            
            # B2B Priority: bypasses minor queues (< 1.05), and takes 0.25x for severe
            severe_b2b = is_b2b & (utils_cong >= 1.05)
            sub_delays[severe_b2b] = 0.25 * base_delays[severe_b2b]
            
            delays[congestion_mask] = sub_delays
            
        return delays

    def generate_shipments(self, n_shipments: int = 50000) -> pd.DataFrame:
        """
        Simulate 50,000 shipments with operational network logic, weather shocks, and SLAs.
        """
        logger.info(f"Generating {n_shipments} synthetic shipment records")
        
        # 1. Select Origin & Destination Hubs (origin != destination)
        origins = self.rng.choice(self.HUBS, size=n_shipments)
        destinations = []
        for o in origins:
            remaining_hubs = [h for h in self.HUBS if h != o]
            destinations.append(self.rng.choice(remaining_hubs))
        destinations = np.array(destinations)

        # 2. Extract distances, base risks, and compute baseline transit times
        distances = np.zeros(n_shipments)
        base_risks = np.zeros(n_shipments)
        for i in range(n_shipments):
            dist, risk = self._get_route_properties(origins[i], destinations[i])
            distances[i] = dist
            base_risks[i] = risk

        # Baseline transit time assuming average speed of 50 km/h
        baseline_transit = distances / 50.0

        # 3. Simulate Priority Tiers (B2B vs C2C)
        # 30% B2B_Priority, 70% C2C_Standard
        shipment_types = self.rng.choice(['B2B_Priority', 'C2C_Standard'], p=[0.3, 0.7], size=n_shipments)

        # 4. Simulate temporal distribution over a 180-day window
        shipment_days = self.rng.integers(0, 180, size=n_shipments)
        start_date = pd.to_datetime('2026-01-01')
        shipment_dates = start_date + pd.to_timedelta(shipment_days, unit='D')
        shipment_dates_str = shipment_dates.strftime('%Y-%m-%d')

        # Monsoon Season Flag (June starts on day 151)
        seasons = np.where(shipment_days >= 151, 'Monsoon', 'Dry')

        # 5. Generate Daily Hub Utilization Rates
        # Pre-generate utilization rate for each hub for each day (60% to 115%)
        daily_utilization = self.rng.uniform(0.60, 1.15, size=(180, len(self.HUBS)))
        hub_to_idx = {hub: idx for idx, hub in enumerate(self.HUBS)}

        # Map daily utilization to each shipment
        orig_utils = np.array([daily_utilization[day, hub_to_idx[o]] for day, o in zip(shipment_days, origins)])
        dest_utils = np.array([daily_utilization[day, hub_to_idx[d]] for day, d in zip(shipment_days, destinations)])

        # 6. Compute Hub Congestion delays (Origin and Destination)
        is_b2b = (shipment_types == 'B2B_Priority')
        orig_congestion_delays = self.compute_congestion_delay(orig_utils, shipment_types)
        dest_congestion_delays = self.compute_congestion_delay(dest_utils, shipment_types)

        # 7. Weather and Route Shock
        # Add slight random noise to base route risk for shipment-level terrain/weather variance
        route_risks = np.clip(base_risks + self.rng.uniform(-0.1, 0.1, size=n_shipments), 0.0, 1.0)
        
        # Base random transit delay (traffic, highway conditions, roadblocks, driver breaks) 
        #gamma because it creates many small delays and few large delays which is what logistic delays looks liek
        base_transit_delays = self.rng.gamma(shape=2.0, scale=1.5, size=n_shipments)

        # Heavy weather shock delay if high risk coincides with monsoon
        weather_multipliers = np.ones(n_shipments)
        monsoon_risk_mask = (seasons == 'Monsoon') & (route_risks > 0.6)
        
        # Multiply transit delay heavily (e.g. scale base delay by 3.5x to 6x based on risk)
        weather_multipliers[monsoon_risk_mask] = (
            self.rng.uniform(3.5, 6.0, size=np.sum(monsoon_risk_mask)) * route_risks[monsoon_risk_mask]
        )
        weather_transit_delays = base_transit_delays * (weather_multipliers - 1)#as base delays would be included later also hence subtract 1 before 

        # 8. compute Total Actual Transit Time includeing actual+total congestion+base random+weather delays.
        actual_transit = (
            baseline_transit +
            orig_congestion_delays +
            dest_congestion_delays +
            base_transit_delays +
            weather_transit_delays
        )

        # 9. compute SLA Windows and Breaches
        # Buffer calibrated to achieve overall breach rate of ~12-15%
        # Target buffer: 1.30 * baseline + 7.0 hours
        # 30 percent accounting for traffic, loading delays, congestions + other delahys
        # 7 hrs accounting for scanning, sorting, dispatching, and loading delays
        sla_windows = baseline_transit * 1.30 + 7.0
        sla_breaches = np.where(actual_transit > sla_windows, 1, 0)


        df = pd.DataFrame({
            'Shipment_ID': [f"SHP_{i:05d}" for i in range(1, n_shipments + 1)],
            'Origin_Hub': origins,
            'Destination_Hub': destinations,
            'Distance_KM': np.round(distances, 1),
            'Baseline_Transit_Time_Hours': np.round(baseline_transit, 2),
            'Shipment_Type': shipment_types,
            'Shipment_Day': shipment_days,
            'Shipment_Date': shipment_dates_str,
            'Season': seasons,
            'Origin_Hub_Utilization': np.round(orig_utils, 3),
            'Destination_Hub_Utilization': np.round(dest_utils, 3),
            'Origin_Congestion_Delay_Hours': np.round(orig_congestion_delays, 2),
            'Destination_Congestion_Delay_Hours': np.round(dest_congestion_delays, 2),
            'Route_Risk_Score': np.round(route_risks, 3),
            'Base_Transit_Delay_Hours': np.round(base_transit_delays, 2),
            'Weather_Transit_Delay_Hours': np.round(weather_transit_delays, 2),
            'Actual_Transit_Time_Hours': np.round(actual_transit, 2),
            'SLA_Window_Hours': np.round(sla_windows, 2),
            'SLA_Breach': sla_breaches
        })

        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save the simulated data to CSV and display summary metrics.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully saved operations dataset to {filepath}")
          
            breach_rate = df['SLA_Breach'].mean()
            logger.info(f"Dataset Shape: {df.shape}")
            logger.info(f"Baseline SLA Breach Rate: {breach_rate:.4%} (Target: 12-15%)")
            
            # Print B2B vs C2C breach rates for operational visibility
            b2b_breaches = df[df['Shipment_Type'] == 'B2B_Priority']['SLA_Breach'].mean()
            c2c_breaches = df[df['Shipment_Type'] == 'C2C_Standard']['SLA_Breach'].mean()
            logger.info(f"B2B Priority SLA Breach Rate: {b2b_breaches:.4%}")
            logger.info(f"C2C Standard SLA Breach Rate: {c2c_breaches:.4%}")
            
            # Print Monsoon vs Dry breach rates for weather visibility
            monsoon_breaches = df[df['Season'] == 'Monsoon']['SLA_Breach'].mean()
            dry_breaches = df[df['Season'] == 'Dry']['SLA_Breach'].mean()
            logger.info(f"Monsoon Season SLA Breach Rate: {monsoon_breaches:.4%}")
            logger.info(f"Dry Season SLA Breach Rate: {dry_breaches:.4%}")
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            raise


if __name__ == "__main__":
    output_path = r"D:\cv_projects_2\c2c_logistics\data\network_operations_data.csv"
    
    generator = OperationsDataGenerator(seed=42)
    df = generator.generate_shipments(n_shipments=50000)
    generator.save_dataset(df, output_path)
