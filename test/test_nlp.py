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

# 1. DATASET PREPARATION
# def load_unitime_dataset(file_path):
#     """Load and preprocess UniTime constraint dataset"""
#     data = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data
def load_unitime_dataset(file_path):
    """Load and preprocess UniTime constraint dataset (supports JSONL format)"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[Warning] Skipped malformed line: {e}")
    return data



def preprocess_function(examples, tokenizer, max_length=512):
    """Preprocess data for FLAN-T5 instruction tuning"""
    inputs = []
    targets = []
    
    # Handle both single example and batch
    if isinstance(examples, dict):
        examples = [examples]
    
    for example in examples:
        # Format input with instruction
        input_text = f"{example['instruction']}\n\nInput: {example['input']}"
        
        # Format output as JSON string
        output_text = json.dumps(example['output'], indent=None, separators=(',', ':'))
        
        inputs.append(input_text)
        targets.append(output_text)
    
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

# 2. MODEL SETUP WITH LORA
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

# 3. TRAINING CONFIGURATION
# 3. TRAINING CONFIGURATION (FIXED)
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
        save_steps=200,                # FIXED: Changed from 500 to 200 (must be multiple of eval_steps)
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False
    )

# 4. CUSTOM METRICS FOR CONSTRAINT ACCURACY
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

# 5. DATASET CREATION HELPER
def create_dataset_from_examples(data, tokenizer):
    """Create HuggingFace dataset from examples"""
    processed_data = []
    
    for example in data:
        processed = preprocess_function(example, tokenizer)
        # Convert tensors to lists for dataset creation
        processed_dict = {
            'input_ids': processed['input_ids'].squeeze().tolist(),
            'attention_mask': processed['attention_mask'].squeeze().tolist(),
            'labels': processed['labels'].squeeze().tolist()
        }
        processed_data.append(processed_dict)
    
    return Dataset.from_list(processed_data)

# 6. MAIN TRAINING FUNCTION
def train_unitime_nlp_model():
    """Complete training pipeline for UniTime NLP constraint extractor"""
    print("*\n****************************************************************")
    print("\nStarting training pipeline for UniTime NLP constraint extractor...")
    print("\nModel and tokenizer initialized with LoRA configuration.")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_lora()

    # Load dataset
    print("*\n****************************************************************")
    print("\nLoading and preprocessing dataset...")
    train_data = load_unitime_dataset(r"/home/sysadm/Music/unitime_nlp/unitime_production_dataset.jsonl")
    eval_data = load_unitime_dataset(r"/home/sysadm/Music/unitime_nlp/eval_dataset.jsonl")
    print(f"Loaded {len(train_data)} training examples and {len(eval_data)} evaluation examples.")


    # Create datasets
    print("*\n****************************************************************")
    print("\nCreating HuggingFace datasets from examples...")
    train_dataset = create_dataset_from_examples(train_data, tokenizer)
    eval_dataset = create_dataset_from_examples(eval_data, tokenizer)
    
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
    trainer.train()
    
    # Save the model
    model.save_pretrained("./unitime-constraint-extractor-final")
    tokenizer.save_pretrained("./unitime-constraint-extractor-final")
    
    print("\nTraining completed and model saved!")
    return model, tokenizer

# 7. INFERENCE FUNCTION
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

# 8. EXAMPLE USAGE AFTER TRAINING
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
    # Train the model
    model, tokenizer = train_unitime_nlp_model()
    
    # Test with example
    test_trained_model()