# # src/data_preprocessing.py
# from datasets import Dataset
# import json

# def convert_to_flan_t5_format(examples):
#     inputs = []
#     targets = []
    
#     for example in examples:
#         # More specific prompts based on type
#         if 'type' in example:
#             type_specific_prompt = {
#                 'reservations': 'Create XML reservation',
#                 'course_offering': 'Generate XML course offering',
#                 'preferences': 'Create XML instructor preferences',
#                 'course_timetable': 'Generate XML course timetable'
#             }
#             prompt_prefix = type_specific_prompt.get(example['type'], 'Convert to XML')
#             input_text = f"{prompt_prefix}: {example['input']}"
#         else:
#             input_text = f"Convert this university scheduling request to XML format: {example['input']}"
        
#         target_text = example['output']  # Keep XML as-is
        
#         inputs.append(input_text)
#         targets.append(target_text)
    
#     return {"input": inputs, "target": targets}


# def prepare_datasets():
#     # Load generated data
#     with open("data/processed/train_dataset.json", "r") as f:
#         train_data = json.load(f)
    
#     with open("data/processed/val_dataset.json", "r") as f:
#         val_data = json.load(f)
    
#     # Convert to FLAN-T5 format
#     train_formatted = convert_to_flan_t5_format(train_data)
#     val_formatted = convert_to_flan_t5_format(val_data)
    
#     # Create Hugging Face datasets
#     train_dataset = Dataset.from_dict(train_formatted)
#     val_dataset = Dataset.from_dict(val_formatted)
    
#     return train_dataset, val_dataset

# data_preprocessing.py
import json
from datasets import Dataset

def load_dataset_from_json(path):
    print(f"Loading dataset from {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {path}")
    return Dataset.from_list(data)

def get_datasets():
    print("Loading all datasets...")
    train_ds = load_dataset_from_json("data/processed/train_dataset.json")
    val_ds = load_dataset_from_json("data/processed/val_dataset.json")
    test_ds = load_dataset_from_json("data/processed/test_dataset.json")
    print("All datasets loaded successfully.")
    return train_ds, val_ds, test_ds
