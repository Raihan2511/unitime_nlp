import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import numpy as np

# 1. FIXED DATASET PREPARATION
def load_unitime_dataset(file_path):
    """Load and preprocess UniTime constraint dataset - FIXED VERSION"""
    data = []
    
    # Determine file format based on extension or content
    if file_path.endswith('.jsonl'):
        # JSONL format - one JSON object per line
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {line[:100]}...")
                        print(f"Error: {e}")
                        continue
    else:
        # JSON format - single JSON object or array
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Try to parse as single JSON object first
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    data = parsed
                else:
                    data = [parsed]
            except json.JSONDecodeError:
                # If single JSON fails, try parsing as multiple JSON objects separated by newlines
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line: {line[:100]}...")
                            print(f"Error: {e}")
                            continue
    
    print(f"Successfully loaded {len(data)} examples from {file_path}")
    return data

def convert_to_jsonl_format(input_file, output_file):
    """Convert your current dataset to proper JSONL format"""
    data = load_unitime_dataset(input_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in data:
            f.write(json.dumps(example, separators=(',', ':')) + '\n')
    
    print(f"Converted {len(data)} examples to JSONL format: {output_file}")

def preprocess_function(examples, tokenizer, max_length=512):
    """Preprocess data for FLAN-T5 instruction tuning - FIXED VERSION"""
    inputs = []
    targets = []
    
    # Handle both single example and batch
    if isinstance(examples, dict) and 'instruction' in examples:
        examples = [examples]
    elif isinstance(examples, list):
        pass
    else:
        # If it's a batch from HuggingFace datasets
        if 'instruction' in examples:
            batch_size = len(examples['instruction'])
            examples = [
                {
                    'instruction': examples['instruction'][i],
                    'input': examples['input'][i],
                    'output': examples['output'][i]
                }
                for i in range(batch_size)
            ]
        else:
            raise ValueError("Unexpected data format in preprocess_function")
    
    for example in examples:
        # Validate example structure
        if not all(key in example for key in ['instruction', 'input', 'output']):
            print(f"Warning: Skipping malformed example: {example}")
            continue
            
        # Format input with instruction
        input_text = f"{example['instruction']}\n\nInput: {example['input']}"
        
        # Format output as JSON string
        if isinstance(example['output'], dict):
            output_text = json.dumps(example['output'], indent=None, separators=(',', ':'))
        elif isinstance(example['output'], str):
            output_text = example['output']
        else:
            output_text = str(example['output'])
        
        inputs.append(input_text)
        targets.append(output_text)
    
    if not inputs:
        raise ValueError("No valid examples found in batch")
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_length, 
        truncation=True, 
        padding=True,
        return_tensors="pt"
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 2. MODEL SETUP WITH LORA (UNCHANGED - LOOKS GOOD)
def setup_model_and_lora():
    """Initialize FLAN-T5 with LoRA configuration"""
    
    # Load base model and tokenizer
    model_name = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # LoRA Configuration - optimized for constraint extraction
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,                           # Low rank dimension
        lora_alpha=32,                  # LoRA scaling parameter
        lora_dropout=0.1,               # Dropout for LoRA layers
        target_modules=[                # Target specific attention layers
            "q", "v", "k", "o",        # Attention projections
            "wi_0", "wi_1", "wo"       # Feed-forward projections
        ],
        bias="none"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    print(f"Trainable parameters: {model.num_parameters()}")
    print(f"Total parameters: {model.num_parameters(only_trainable=False)}")
    
    return model, tokenizer

# 3. TRAINING CONFIGURATION (UNCHANGED - LOOKS GOOD)
def setup_training_args():
    """Configure training parameters for UniTime constraint extraction"""
    
    return TrainingArguments(
        output_dir="./unitime-constraint-extractor",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        learning_rate=1e-4,
        fp16=True,                     # Mixed precision training
        logging_steps=50,
        eval_steps=200,
        save_steps=500,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False
    )

# 4. CUSTOM METRICS FOR CONSTRAINT ACCURACY (UNCHANGED - LOOKS GOOD)
def compute_constraint_metrics(eval_pred, tokenizer):
    """Custom metrics for constraint extraction accuracy"""
    predictions, labels = eval_pred
    
    # Handle padding tokens
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Parse JSON and calculate accuracy
    exact_matches = 0
    constraint_accuracy = 0
    total_constraints = 0
    valid_jsons = 0
    
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            pred_json = json.loads(pred)
            valid_jsons += 1
            
            try:
                label_json = json.loads(label)
                
                # Exact match accuracy
                if pred_json == label_json:
                    exact_matches += 1
                
                # Constraint-level accuracy
                pred_constraints = pred_json.get('constraints', [])
                label_constraints = label_json.get('constraints', [])
                
                total_constraints += len(label_constraints)
                
                for label_constraint in label_constraints:
                    for pred_constraint in pred_constraints:
                        if (pred_constraint.get('type') == label_constraint.get('type') and 
                            pred_constraint.get('entity_name') == label_constraint.get('entity_name')):
                            constraint_accuracy += 1
                            break
                            
            except json.JSONDecodeError:
                continue
                
        except json.JSONDecodeError:
            continue
    
    return {
        "exact_match_accuracy": exact_matches / len(decoded_preds),
        "constraint_accuracy": constraint_accuracy / max(total_constraints, 1),
        "valid_json_ratio": valid_jsons / len(decoded_preds)
    }

# 5. FIXED DATASET CREATION HELPER
def create_dataset_from_examples(data, tokenizer):
    """Create HuggingFace dataset from examples - FIXED VERSION"""
    
    # First, create the raw dataset
    dataset_dict = {
        'instruction': [],
        'input': [],
        'output': []
    }
    
    for example in data:
        if all(key in example for key in ['instruction', 'input', 'output']):
            dataset_dict['instruction'].append(example['instruction'])
            dataset_dict['input'].append(example['input'])
            dataset_dict['output'].append(example['output'])
        else:
            print(f"Warning: Skipping malformed example: {example}")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Apply preprocessing
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer)
    
    # Process the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

# 6. MAIN TRAINING FUNCTION (UPDATED WITH BETTER ERROR HANDLING)
def train_unitime_nlp_model():
    """Complete training pipeline for UniTime NLP constraint extractor"""
    print("*\n****************************************************************")
    print("\nStarting training pipeline for UniTime NLP constraint extractor...")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_lora()
    print("\nModel and tokenizer initialized with LoRA configuration.")

    # Load dataset
    print("*\n****************************************************************")
    print("\nLoading and preprocessing dataset...")
    
    # Update these paths to your actual file paths
    train_file = "paste.txt"  # Your training data file
    eval_file = "paste.txt"   # You might want to split this or create a separate eval file
    
    try:
        train_data = load_unitime_dataset(train_file)
        
        # For now, use the same data for eval (you should create a separate eval set)
        # Split the data: 80% train, 20% eval
        split_idx = int(len(train_data) * 0.8)
        eval_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
        
        print(f"Loaded {len(train_data)} training examples and {len(eval_data)} evaluation examples.")
        
        # Optionally, convert to JSONL format for future use
        # convert_to_jsonl_format(train_file, "unitime_training_data.jsonl")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    # Create datasets
    print("*\n****************************************************************")
    print("\nCreating HuggingFace datasets from examples...")
    
    try:
        train_dataset = create_dataset_from_examples(train_data, tokenizer)
        eval_dataset = create_dataset_from_examples(eval_data, tokenizer)
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return None, None
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Training arguments
    training_args = setup_training_args()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_constraint_metrics(eval_pred, tokenizer)
    )
    
    # Start training
    print("*\n****************************************************************")
    print("\nStarting training...")
    try:
        trainer.train()
        
        # Save the model
        model.save_pretrained("./unitime-constraint-extractor-final")
        tokenizer.save_pretrained("./unitime-constraint-extractor-final")
        
        print("\nTraining completed and model saved!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None

# 7. INFERENCE FUNCTION (UNCHANGED - LOOKS GOOD)
def extract_constraints(input_text, model, tokenizer):
    """Extract constraints from natural language input"""
    
    instruction = "Extract scheduling constraints from the following academic timetabling request and format them as structured JSON for UniTime database insertion:"
    
    full_input = f"{instruction}\n\nInput: {input_text}"
    
    # Tokenize input
    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=1024,
            num_beams=4,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and parse response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        constraints = json.loads(response)
        return constraints
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response", "raw_output": response}

# 8. UTILITY FUNCTION TO VALIDATE DATASET
def validate_dataset(file_path):
    """Validate dataset structure and content"""
    print(f"Validating dataset: {file_path}")
    
    try:
        data = load_unitime_dataset(file_path)
        print(f"Total examples loaded: {len(data)}")
        
        # Check structure
        required_keys = ['instruction', 'input', 'output']
        valid_examples = 0
        
        for i, example in enumerate(data[:5]):  # Check first 5 examples
            print(f"\nExample {i+1}:")
            print(f"  Keys: {list(example.keys())}")
            
            if all(key in example for key in required_keys):
                valid_examples += 1
                print(f"  ✓ Valid structure")
                print(f"  Instruction: {example['instruction'][:100]}...")
                print(f"  Input: {example['input']}")
                print(f"  Output type: {type(example['output'])}")
            else:
                print(f"  ✗ Missing required keys")
        
        print(f"\nValid examples in sample: {valid_examples}/5")
        return data
        
    except Exception as e:
        print(f"Error validating dataset: {e}")
        return None

# 9. EXAMPLE USAGE AFTER TRAINING (UNCHANGED)
def test_trained_model():
    """Test the trained model with example inputs"""
    
    # Load trained model
    model = T5ForConditionalGeneration.from_pretrained("./unitime-constraint-extractor-final")
    tokenizer = T5Tokenizer.from_pretrained("./unitime-constraint-extractor-final")
    
    # Test input
    test_input = "Professor Davis prefers teaching on Mondays and Wednesdays in the morning before 11 AM"
    
    # Extract constraints
    result = extract_constraints(test_input, model, tokenizer)
    
    print("Input:", test_input)
    print("Output:", json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    # First validate your dataset
    print("Validating dataset...")
    data = validate_dataset("paste.txt")
    
    if data:
        print("\nDataset validation successful. Starting training...")
        # Train the model
        model, tokenizer = train_unitime_nlp_model()
        
        if model and tokenizer:
            # Test with example
            test_trained_model()
        else:
            print("Training failed!")
    else:
        print("Dataset validation failed!")