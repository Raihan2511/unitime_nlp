import json
import os

def merge_test_data(offer_path, reserv_path, pref_path, output_path):
    merged = []

    # Load each dataset
    for path in [offer_path, reserv_path, pref_path]:
        with open(path, "r") as f:
            data = json.load(f)
            merged.extend(data)

    # Save to output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"✅ Merged test data saved to {output_path}")

# This script merges the test datasets from offerings, reservations, and preferences into a single JSON file.
if __name__ == "__main__":
    merge_test_data(
        "data/offerings_data/test_offer.json",
        "data/reservation_data/test_reservation.json",
        "data/preference_data/test_pref.json",
        "data/processed/test_data.json"
    )