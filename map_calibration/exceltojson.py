# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:42:23 2025

@author: carbse_ra10
"""

import pandas as pd
import json

# Load your Excel file
df = pd.read_excel("map_lat_long.xlsx")  # ← Change filename if needed

# Make sure the columns are exactly: 'City', 'Lat', 'Long'
if not all(col in df.columns for col in ['City', 'Lat', 'Long']):
    raise ValueError("Excel must have 'City', 'Lat', 'Long' columns")

# Convert to desired JSON structure
city_json = {
    row['City']: {"Lat": row['Lat'], "Long": row['Long']}
    for _, row in df.iterrows()
}

# Save to file
with open("city_metadata.json", "w", encoding="utf-8") as f:
    json.dump(city_json, f, indent=2)

print("✔ city_metadata.json created.")
