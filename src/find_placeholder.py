import json
import re
import os

def detect_placeholders(offerings_file, reservations_file, preferences_file):
    """Detect placeholders from different dataset files and save to JSON"""
    
    def extract_placeholders(text):
        """Extract placeholders like [PLACEHOLDER] from text"""
        return re.findall(r'\[([^\]]+)\]', text)
    
    def find_dataset_placeholders(file_path, dataset_name):
        """Find all unique placeholders in a dataset file"""
        placeholders = set()
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                input_text = item.get("input", "")
                output_text = item.get("output", "")
                placeholders.update(extract_placeholders(input_text))
                placeholders.update(extract_placeholders(output_text))
                
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            return set()
        
        return sorted(list(placeholders))
    
    # Detect placeholders for each dataset
    placeholders_data = {
        "preferences": find_dataset_placeholders(preferences_file, "preferences"),
        "reservations": find_dataset_placeholders(reservations_file, "reservations"),
        "offerings": find_dataset_placeholders(offerings_file, "offerings")
    }
    
    # Create directory if it doesn't exist
    os.makedirs("data/placeholder", exist_ok=True)
    
    # Save to JSON file
    with open("data/placeholder/placeholders.json", "w") as f:
        json.dump(placeholders_data, f, indent=2)
    
    print("Placeholders detected and saved to data/placeholder/placeholders.json")
    print("\nDetected placeholders:")
    for dataset, placeholders in placeholders_data.items():
        print(f"{dataset}: {placeholders}")
    
    return placeholders_data

if __name__ == "__main__":
    # File paths
    offerings_file = "data/offerings_data/train_offer.json"
    reservations_file = "data/reservation_data/train_reservation.json" 
    preferences_file = "data/preference_data/train_pref.json"
    
    detect_placeholders(offerings_file, reservations_file, preferences_file)