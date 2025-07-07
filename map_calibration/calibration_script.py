# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:25:36 2025

@author: carbse_ra10
"""

import json
import joblib
from PIL import Image
import numpy as np
from sklearn.linear_model import LinearRegression

# ─── CONFIG ─────────────────────────────────────────────────────
MAP_IMAGE_PATH = "india-map.jpg"           # Your base image
CITY_META_FILE = "city_metadata.json"      # File with all cities and lat/lon
MODEL_FILE = "map_model.pkl"               # Output: saved model
PIXEL_OUTPUT = "city_pixel_positions.json" # Output: pixel positions
# ────────────────────────────────────────────────────────────────

# Calibration data: Add accurate pixel (x, y) manually from your image editor
CALIBRATION_CITIES = {
    "New Delhi":     {"Lat": 28.61, "Lon": 77.20, "x": 192, "y": 191},
    "Mumbai":    {"Lat": 19.07, "Lon": 72.87, "x": 91, "y": 394},
    "Chennai":   {"Lat": 13.08, "Lon": 80.27, "x": 232, "y": 562},
    "Kolkata":   {"Lat": 22.57, "Lon": 88.36, "x": 414, "y": 329},
    "Ahmedabad": {"Lat": 23.02, "Lon": 72.57, "x": 86, "y": 302},
    "Bengaluru":  {"Lat": 12.97, "Lon": 77.59, "x": 168, "y": 571},
    "Panaji":  {"Lat": 15.48, "Lon": 73.83, "x": 414, "y": 329},
}

# ─── TRAINING ────────────────────────────────────────────────────
def train_model(calib):
    coords = np.array([[v["Lat"], v["Lon"]] for v in calib.values()])
    x_vals = np.array([v["x"] for v in calib.values()])
    y_vals = np.array([v["y"] for v in calib.values()])

    model_x = LinearRegression().fit(coords, x_vals)
    model_y = LinearRegression().fit(coords, y_vals)

    return model_x, model_y

# ─── LOAD CITY DATA ──────────────────────────────────────────────
def load_city_meta(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── RUN ─────────────────────────────────────────────────────────
def main():
    model_x, model_y = train_model(CALIBRATION_CITIES)

    # Save models to file
    joblib.dump((model_x, model_y), MODEL_FILE)
    print(f"✔ Models saved to {MODEL_FILE}")

    # Predict positions for all cities
    city_meta = load_city_meta(CITY_META_FILE)
    pixel_positions = {}

    for city, meta in city_meta.items():
        lat, lon = meta["Lat"], meta["Long"]
        x = int(round(model_x.predict([[lat, lon]])[0]))
        y = int(round(model_y.predict([[lat, lon]])[0]))
        pixel_positions[city] = {"x": x, "y": y, "Lat": lat, "Long": lon}

    with open(PIXEL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(pixel_positions, f, indent=2)

    print(f"✔ Pixel positions written to {PIXEL_OUTPUT}")

if __name__ == "__main__":
    main()
