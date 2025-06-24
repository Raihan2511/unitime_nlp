# main.py
from src.load_data import load_train_datasets
from src.load_data import load_val_datasets
from src.load_data import load_test_datasets
from src.data_preprocessing import preprocess_data
from src.data_preprocessing import merge_test_data

from src.train_model import train_model
from src.evaluation import evaluate_model

import json

# Load training datasets 
offerings, reservations, preferences,timetbale = load_train_datasets()
train_processed_data = preprocess_data(offerings, reservations, preferences, timetbale)

# loading validation datasets 
val_offerings, val_reservations, val_preferences,val_timetable = load_val_datasets()
processed_val = preprocess_data(val_offerings, val_reservations, val_preferences, val_timetable)

# Load test datasets
test_offerings, test_reservations, test_preferences,test_timetable = load_test_datasets()
merge_test_data(
    test_offerings,
    test_reservations,
    test_preferences,
    test_timetable,
    "data/processed/test_data.json"
)
print(f"✅ Loaded {len(train_processed_data)} training and {len(processed_val)} validation samples.")

# Save the full processed dataset to file (no splitting)
with open("data/processed/train_data.json", "w") as f:
    json.dump(train_processed_data, f, indent=2)

with open("data/processed/val_data.json", "w") as f:
    json.dump(processed_val, f, indent=2)


# with open("data/processed/test_data.json", "w") as f:
#     json.dump(test_processed, f, indent=2)

# {len(test_processed)} test

print(f"✅ Saved {len(train_processed_data)} training and {len(processed_val)} and  samples to JSON files.")
print("\n✅ Saved training, validation, and test datasets to JSON files.")

# Start training with validation
print("\nStarting training with validation...")
train_model("data/processed/train_data.json", "data/processed/val_data.json")

print("\nStarting evaluation on test data...")
evaluate_model("./models/final_model", "data/processed/test_data.json")
