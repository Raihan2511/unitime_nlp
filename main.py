# main.py
from src.load_data import load_train_datasets
from src.load_data import load_val_datasets
from src.data_preprocessing import preprocess_data
from src.train_model import train_model
import json

# Load training datasets 
offerings, reservations, preferences = load_train_datasets()
train_processed_data = preprocess_data(offerings, reservations, preferences)

# loading validation datasets 
val_offerings, val_reservations, val_preferences = load_val_datasets()
processed_val = preprocess_data(val_offerings, val_reservations, val_preferences)




# Save the full processed dataset to file (no splitting)
with open("data/processed/train_data.json", "w") as f:
    json.dump(train_processed_data, f, indent=2)

# print(f"✅ Saved {len(train_processed_data)} samples to data/processed/train_data.json")

# Start training
# train_model("data/processed/train_data.json")


with open("data/processed/val_data.json", "w") as f:
    json.dump(processed_val, f, indent=2)

print(f"✅ Saved {len(train_processed_data)} training and {len(processed_val)} validation samples.")

# Start training with validation
train_model("data/processed/train_data.json", "data/processed/val_data.json")
