# main.py
from src.load_data import load_datasets
from src.data_preprocessing import preprocess_data
import json

offerings, reservations, preferences = load_datasets()
processed_data = preprocess_data(offerings, reservations, preferences)

with open("train_data.json", "w") as f:
    json.dump(processed_data, f, indent=2)

print(f"Generated {len(processed_data)} training samples and saved to train_data.json")