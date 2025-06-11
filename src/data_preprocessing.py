# src/data_preprocessing.py
from datasets import Dataset
import json

def convert_to_flan_t5_format(examples):
    """Convert our dataset to FLAN-T5 input-output format"""
    inputs = []
    targets = []
    
    for example in examples:
        # Create instruction-based input
        input_text = f"Convert this scheduling request to structured format: {example['input']}"
        
        # Create JSON output as target
        target_text = json.dumps(example['output'], ensure_ascii=False)
        
        inputs.append(input_text)
        targets.append(target_text)
    
    return {"input": inputs, "target": targets}

def prepare_datasets():
    # Load generated data
    with open("data/processed/train_dataset.json", "r") as f:
        train_data = json.load(f)
    
    with open("data/processed/val_dataset.json", "r") as f:
        val_data = json.load(f)
    
    # Convert to FLAN-T5 format
    train_formatted = convert_to_flan_t5_format(train_data)
    val_formatted = convert_to_flan_t5_format(val_data)
    
    # Create Hugging Face datasets
    train_dataset = Dataset.from_dict(train_formatted)
    val_dataset = Dataset.from_dict(val_formatted)
    
    return train_dataset, val_dataset