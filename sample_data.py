import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data() -> pd.DataFrame:
    """
    Generate 17 months of highly realistic sales data for an Indian grocery/retail shop.
    Returns:
        pd.DataFrame: A clean DataFrame with the standard Indian retail columns:
                     - Date
                     - Product
                     - Quantity Sold
                     - Revenue (₹)
                     - Category
                     - Unit Price (₹)
    """
    np.random.seed(42)
    
    # 17 months range: Jan 1, 2025 to May 31, 2026
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 5, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Product catalog with category, pricing, and relative popularity
    # Popularity dictates average daily sales volume
    catalog = [
        # Staples (High Value, Regular Demand)
        {"Product": "Basmati Rice 5kg", "Category": "Staples", "UnitPrice": 380.0, "Cost": 300.0, "BaseQty": 4.5},
        {"Product": "Ashirvaad Atta 5kg", "Category": "Staples", "UnitPrice": 240.0, "Cost": 190.0, "BaseQty": 6.0},
        {"Product": "Fortune Mustard Oil 1L", "Category": "Staples", "UnitPrice": 175.0, "Cost": 140.0, "BaseQty": 5.0},
        {"Product": "Tata Salt Lite 1kg", "Category": "Staples", "UnitPrice": 30.0, "Cost": 22.0, "BaseQty": 7.0},
        
        # Dairy (High Volume, Short Shelf Life)
        {"Product": "Amul Butter 500g", "Category": "Dairy", "UnitPrice": 260.0, "Cost": 210.0, "BaseQty": 3.8},
        {"Product": "Amul Gold Milk 1L", "Category": "Dairy", "UnitPrice": 66.0, "Cost": 54.0, "BaseQty": 12.0},
        {"Product": "Mother Dairy Paneer 200g", "Category": "Dairy", "UnitPrice": 90.0, "Cost": 72.0, "BaseQty": 2.5},
        
        # Beverages
        {"Product": "Tata Tea Gold 500g", "Category": "Beverages", "UnitPrice": 320.0, "Cost": 250.0, "BaseQty": 2.2},
        {"Product": "Nescafe Classic 100g", "Category": "Beverages", "UnitPrice": 360.0, "Cost": 290.0, "BaseQty": 1.5},
        
        # Packaged Foods
        {"Product": "Maggi 2-Min Noodles 12pk", "Category": "Packaged Foods", "UnitPrice": 168.0, "Cost": 135.0, "BaseQty": 5.5},
        {"Product": "Britannia Marie Gold 250g", "Category": "Packaged Foods", "UnitPrice": 35.0, "Cost": 27.0, "BaseQty": 6.5},
        {"Product": "Haldiram Bhujia Sev 400g", "Category": "Packaged Foods", "UnitPrice": 110.0, "Cost": 85.0, "BaseQty": 3.2},
        
        # Personal Care & Home Care (Medium-to-Low Volume)
        {"Product": "Dettol Liquid Handwash 900ml", "Category": "Personal Care", "UnitPrice": 99.0, "Cost": 78.0, "BaseQty": 1.8},
        {"Product": "Colgate MaxFresh 150g", "Category": "Personal Care", "UnitPrice": 75.0, "Cost": 58.0, "BaseQty": 2.1},
        {"Product": "Surf Excel Easy Wash 1kg", "Category": "Home Care", "UnitPrice": 140.0, "Cost": 112.0, "BaseQty": 2.8},
        {"Product": "Vim Dishwash Gel 500ml", "Category": "Home Care", "UnitPrice": 99.0, "Cost": 80.0, "BaseQty": 3.0},
        
        # Slow-Moving Specialty Items (To trigger slow-mover alerts)
        {"Product": "Organic Quinoa 500g", "Category": "Staples", "UnitPrice": 450.0, "Cost": 350.0, "BaseQty": 0.15},
        {"Product": "Diet Cranberry Juice 1L", "Category": "Beverages", "UnitPrice": 220.0, "Cost": 170.0, "BaseQty": 0.22},
        {"Product": "Saffron (Kesar) 1g", "Category": "Staples", "UnitPrice": 399.0, "Cost": 310.0, "BaseQty": 0.08},
    ]
    
    rows = []
    
    for date in date_range:
        # 1. Weekly pattern: Saturday and Sunday have higher sales (1.4x), Midweek Wednesday has a slight dip (0.9x)
        dow = date.dayofweek
        if dow in [5, 6]:  # Sat, Sun
            weekly_mult = 1.35 + np.random.uniform(-0.05, 0.05)
        elif dow == 2:     # Wed
            weekly_mult = 0.88 + np.random.uniform(-0.04, 0.04)
        else:
            weekly_mult = 1.0 + np.random.uniform(-0.05, 0.05)
            
        # 2. Monthly Growth Trend: 1.2% compound growth monthly
        months_since_start = (date.year - start_date.year) * 12 + (date.month - start_date.month)
        growth_mult = (1.012) ** months_since_start
        
        # 3. Seasonal & Holiday Spikes:
        # - Diwali Peak (around mid-November 2025): major spike for staples, saffron, bhujia, etc.
        # - Holi Spike (around mid-March): sweets, beverages
        # - General Summer increase in Dairy/Beverages (Milk, paneer, juice)
        seasonal_mult = 1.0
        
        # Diwali spike (Nov 5 - Nov 12, 2025)
        if date >= datetime(2025, 11, 1) and date <= datetime(2025, 11, 12):
            seasonal_mult = 2.2 if date.day >= 8 else 1.5
        # Holi spike (March 10 - March 15, 2026)
        elif date >= datetime(2026, 3, 10) and date <= datetime(2026, 3, 15):
            seasonal_mult = 1.4
            
        # For each product, simulate daily quantity sold
        for prod in catalog:
            base_qty = prod["BaseQty"]
            
            # Apply adjustments
            daily_popularity_fluct = np.random.normal(1.0, 0.25)
            
            # Adjust slow-moving items to be highly sparse (zero on most days)
            if base_qty < 0.5:
                # Bernoulli trial: does a sale occur today?
                prob_of_sale = base_qty * weekly_mult * growth_mult
                sold = 1 if np.random.uniform(0, 1) < prob_of_sale else 0
                qty = float(sold)
            else:
                qty = base_qty * weekly_mult * growth_mult * seasonal_mult * daily_popularity_fluct
                qty = max(0.0, qty)
                qty = float(np.round(qty))
            
            # If quantity is sold, generate sales record
            if qty > 0:
                revenue = qty * prod["UnitPrice"]
                rows.append({
                    "Date": date,
                    "Product": prod["Product"],
                    "Quantity Sold": qty,
                    "Revenue (₹)": revenue,
                    "Category": prod["Category"],
                    "Unit Price (₹)": prod["UnitPrice"]
                })
                
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Date", "Product"]).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_sample_data()
    print(f"Generated {len(df)} rows of retail data.")
    print(df.head())
