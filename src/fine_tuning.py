# # src/fine_tuning.py - OPTIMIZED VERSION FOR XML DATASET
# import torch
# from transformers import (
#     T5ForConditionalGeneration, 
#     T5Tokenizer, 
#     Trainer, 
#     TrainingArguments,
#     DataCollatorForSeq2Seq
# )
# from datasets import Dataset
# import json
# import os
# import xml.etree.ElementTree as ET
# from collections import Counter

# class UniTimeFlanT5Trainer:
#     def __init__(self, model_name="google/flan-t5-small"):  # Changed to base for better XML handling
#         self.model_name = model_name
#         self.tokenizer = T5Tokenizer.from_pretrained(model_name)
#         self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
#         # Add XML-specific tokens
#         xml_tokens = [
#             "<reservations>", "</reservations>", "<reservation>", "</reservation>",
#             "<offerings>", "</offerings>", "<offering>", "</offering>",
#             "<preferences>", "</preferences>", "<instructor>", "</instructor>",
#             "<timetable>", "</timetable>", "<class>", "</class>",
#             "<course>", "</course>", "<time>", "</time>", "<room>", "</room>",
#             "<timePref>", "</timePref>", "<pref>", "</pref>",
#             "<academicClassification>", "</academicClassification>",
#             "<major>", "</major>", "<arrangeTime>", "</arrangeTime>"
#         ]
        
#         # Add the new tokens
#         num_added = self.tokenizer.add_tokens(xml_tokens)
#         print(f"Added {num_added} XML tokens to tokenizer")
        
#         # Resize model embeddings
#         self.model.resize_token_embeddings(len(self.tokenizer))

#     def analyze_dataset(self, dataset):
#         """Analyze the dataset to understand XML structure patterns"""
#         print("\n📊 Dataset Analysis:")
        
#         # Count by type
#         type_counts = Counter()
#         xml_lengths = []
        
#         for example in dataset:
#             if isinstance(example, dict) and 'type' in example:
#                 type_counts[example['type']] += 1
            
#             # Analyze XML length
#             if 'output' in example:
#                 xml_lengths.append(len(example['output']))
        
#         print(f"Dataset size: {len(dataset)}")
#         print("Type distribution:")
#         for type_name, count in type_counts.items():
#             print(f"  {type_name}: {count} ({count/len(dataset)*100:.1f}%)")
        
#         if xml_lengths:
#             print(f"XML output lengths:")
#             print(f"  Average: {sum(xml_lengths)/len(xml_lengths):.0f} chars")
#             print(f"  Max: {max(xml_lengths)} chars")
#             print(f"  Min: {min(xml_lengths)} chars")
        
#         return type_counts

#     def preprocess_function(self, examples):
#         """Enhanced preprocessing for XML data"""
#         inputs = examples["input"]
#         targets = examples["target"]
        
#         # Create more specific prompts based on the task type
#         enhanced_inputs = []
#         for inp in inputs:
#             # Add XML formatting instruction
#             enhanced_input = f"Convert this university scheduling request to XML format: {inp}"
#             enhanced_inputs.append(enhanced_input)
        
#         # Tokenize inputs with longer max length for XML
#         model_inputs = self.tokenizer(
#             enhanced_inputs, 
#             max_length=512,  # Increased for XML complexity
#             truncation=True, 
#             padding="max_length"
#         )
        
#         # Tokenize XML targets
#         with self.tokenizer.as_target_tokenizer():
#             labels = self.tokenizer(
#                 targets, 
#                 max_length=512,  # Increased for XML complexity
#                 truncation=True, 
#                 padding="max_length"
#             )
        
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs


#     # def preprocess_function(self, examples):
#     #     """Enhanced preprocessing for XML data"""
#     #     inputs = examples["input"]
#     #     targets = examples["target"]
        
#     #     # Create more specific prompts based on the task type
#     #     enhanced_inputs = []
#     #     for inp in inputs:
#     #         enhanced_input = f"Convert this university scheduling request to XML format: {inp}"
#     #         enhanced_inputs.append(enhanced_input)
        
#     #     # Tokenize inputs with longer max length for XML
#     #     model_inputs = self.tokenizer(
#     #         enhanced_inputs, 
#     #         max_length=512,
#     #         truncation=True,
#     #         padding="max_length"
#     #     )
        
#     #     # Tokenize XML targets
#     #     with self.tokenizer.as_target_tokenizer():
#     #         labels = self.tokenizer(
#     #             targets, 
#     #             max_length=512,
#     #             truncation=True,
#     #             padding="max_length"
#     #         )
        
#     #     model_inputs["labels"] = labels["input_ids"]
        
#     #     # 🔍 Print token lengths for debugging
#     #     for i in range(len(enhanced_inputs)):
#     #         input_tokens = self.tokenizer.tokenize(enhanced_inputs[i])
#     #         target_tokens = self.tokenizer.tokenize(targets[i])
            
#     #         print(f"\n📏 Sample {i + 1}")
#     #         print(f"Input text: {enhanced_inputs[i][:80]}...")
#     #         print(f"➡️ Input tokens: {len(input_tokens)}")
#     #         print(f"Target text: {targets[i][:80]}...")
#     #         print(f"➡️ Target tokens: {len(target_tokens)}")
            
#     #         if len(input_tokens) > 512 or len(target_tokens) > 512:
#     #             print("⚠️ Warning: Token length exceeds 512! May be truncated.")

#     #     return model_inputs


#     def validate_xml_samples(self, dataset, num_samples=5):
#         """Validate XML structure in samples"""
#         print(f"\n🔍 Validating XML structure in {num_samples} samples:")
        
#         valid_count = 0
#         for i, example in enumerate(dataset[:num_samples]):
#             if 'output' in example:
#                 try:
#                     ET.fromstring(example['output'])
#                     print(f"✅ Sample {i+1}: Valid XML")
#                     valid_count += 1
#                 except ET.ParseError as e:
#                     print(f"❌ Sample {i+1}: Invalid XML - {e}")
#                     print(f"   Content: {example['output'][:100]}...")
        
#         print(f"Valid XML samples: {valid_count}/{num_samples}")
#         return valid_count == num_samples

#     def fine_tune(self, train_dataset, val_dataset, output_dir="models/flan_t5_unitime_xml"):
#         """Fine-tune the model for XML generation"""
        
#         # Create output directory
#         os.makedirs(output_dir, exist_ok=True)
#         os.makedirs(f"{output_dir}/logs", exist_ok=True)
        
#         print(f"Output directory: {os.path.abspath(output_dir)}")
        
#         # Analyze dataset
#         train_raw = []
#         val_raw = []
        
#         # Convert Dataset objects to lists for analysis
#         if hasattr(train_dataset, '__iter__'):
#             for item in train_dataset:
#                 train_raw.append(item)
#         if hasattr(val_dataset, '__iter__'):
#             for item in val_dataset:
#                 val_raw.append(item)
        
#         # Validate XML samples
#         print("Validating training samples...")
#         # Note: This assumes your dataset has 'output' field
        
#         # Tokenize datasets
#         print("Tokenizing datasets...")
#         train_dataset = train_dataset.map(
#             self.preprocess_function, 
#             batched=True, 
#             remove_columns=train_dataset.column_names
#         )
        
#         val_dataset = val_dataset.map(
#             self.preprocess_function, 
#             batched=True, 
#             remove_columns=val_dataset.column_names
#         )
        
#         # Data collator
#         data_collator = DataCollatorForSeq2Seq(
#             tokenizer=self.tokenizer,
#             model=self.model,
#             padding=True
#         )
        
#         # Optimized training arguments for XML generation
#         training_args = TrainingArguments(
#             output_dir=output_dir,
#             num_train_epochs=10,  # More epochs for complex XML structure
#             per_device_train_batch_size=2,  # Smaller batch due to longer sequences
#             per_device_eval_batch_size=2,
#             learning_rate=3e-4,  # Slightly lower learning rate for stability
#             weight_decay=0.01,
#             warmup_steps=200,  # More warmup steps
#             logging_dir=f"{output_dir}/logs",
#             logging_steps=10,
#             eval_strategy="steps",
#             eval_steps=50,
#             save_steps=100,
#             save_total_limit=5,  # Keep more checkpoints
#             load_best_model_at_end=True,
#             metric_for_best_model="eval_loss",
#             greater_is_better=False,
#             push_to_hub=False,
#             report_to=["tensorboard"],
#             fp16=False,  # Disable for stability with complex XML
#             dataloader_pin_memory=False,
#             remove_unused_columns=False,
#             gradient_accumulation_steps=8,  # Larger effective batch size
#             adam_epsilon=1e-8,
#             max_grad_norm=1.0,
#             lr_scheduler_type="cosine",  # Better for longer training
#             save_safetensors=False,
#             # XML-specific optimizations
#             dataloader_num_workers=0,  # Avoid multiprocessing issues
#             prediction_loss_only=False,
#         )

#         # Trainer
#         trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=val_dataset,
#             data_collator=data_collator,
#         )
        
#         # Train
#         print("Starting XML-focused training...")
#         try:
#             trainer.train()
            
#             # Save model
#             print("Saving model...")
#             trainer.save_model(output_dir)
#             # self.tokenizer.save_pretrained(output_dir)
#             self.tokenizer.save_pretrained(output_dir)

#             # Manually save tokenizer.json if missing
#             # if self.tokenizer.is_fast and not os.path.exists(os.path.join(output_dir, "tokenizer.json")):
#             #     self.tokenizer.backend_tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

            
#             # Explicit backup save
#             torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
#             self.model.save_pretrained(output_dir, safe_serialization=False)
            
#             # Verify files
#             required_files = [
#                 "config.json", 
#                 "pytorch_model.bin", 
#                 "tokenizer_config.json",
#                 "tokenizer.json"
#             ]
            
#             print("\nVerifying saved files:")
#             all_files_exist = True
#             for file in required_files:
#                 file_path = os.path.join(output_dir, file)
#                 if os.path.exists(file_path):
#                     file_size = os.path.getsize(file_path)
#                     print(f"✅ {file} saved successfully ({file_size:,} bytes)")
#                 else:
#                     print(f"❌ {file} not found!")
#                     all_files_exist = False
            
#             # Test inference on a sample
#             print("\n🧪 Testing inference on sample:")
#             test_input = "Reserve 20 seats in CS101 for COMP majors"
#             result = self.test_inference(test_input)
#             print(f"Input: {test_input}")
#             print(f"Output: {result}")
            
#             # Create metadata
#             metadata = {
#                 "model_name": self.model_name,
#                 "training_examples": len(train_dataset),
#                 "validation_examples": len(val_dataset),
#                 "epochs": training_args.num_train_epochs,
#                 "batch_size": training_args.per_device_train_batch_size,
#                 "learning_rate": training_args.learning_rate,
#                 "max_length": 512,
#                 "status": "completed",
#                 "files_verified": all_files_exist,
#                 "xml_focused": True
#             }
            
#             with open(f"{output_dir}/model_info.json", "w") as f:
#                 json.dump(metadata, f, indent=2)
            
#             return trainer
            
#         except Exception as e:
#             print(f"Training failed with error: {e}")
#             error_info = {
#                 "status": "failed",
#                 "error": str(e),
#                 "model_name": self.model_name
#             }
#             with open(f"{output_dir}/error_log.json", "w") as f:
#                 json.dump(error_info, f, indent=2)
#             raise

#     def test_inference(self, input_text):
#         """Test inference on a single input"""
#         prompt = f"Convert this university scheduling request to XML format: {input_text}"
#         inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

#         # Get device
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(device)
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_length=512,
#                 num_beams=4,
#                 temperature=0.7,
#                 do_sample=True,
#                 pad_token_id=self.tokenizer.eos_token_id,
#                 early_stopping=True
#             )

#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# # Training script
# if __name__ == "__main__":
#     from data_preprocessing import prepare_datasets
    
#     print("🚀 Starting XML-focused fine-tuning...")
#     print("Preparing datasets...")
    
#     # Prepare datasets
#     train_dataset, val_dataset = prepare_datasets()
    
#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
    
#     # Show samples
#     print("\nDataset samples:")
#     print(f"Train sample: {train_dataset[0]}")
    
#     # Initialize trainer with base model for better XML handling
#     trainer = UniTimeFlanT5Trainer("google/flan-t5-small")
    
#     # Fine-tune
#     try:
#         trained_model = trainer.fine_tune(train_dataset, val_dataset)
#         print("✅ XML-focused fine-tuning completed successfully!")
#     except Exception as e:
#         print(f"❌ Fine-tuning failed: {e}")
#         import traceback
#         traceback.print_exc()

# src/fine_tuning.py - FIXED VERSION FOR XML DATASET

# finetuning.py
import os
os.environ["WANDB_DISABLED"] = "true"
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from data_preprocessing import get_datasets

model_name = "google/flan-t5-large"
print(f"Loading model and tokenizer: {model_name}")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
print("Model and tokenizer loaded.")

def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Fetching datasets...")
train_dataset, val_dataset, test_dataset = get_datasets()

print("Preprocessing training dataset...")
train_dataset = train_dataset.map(preprocess_function, batched=True)
print("Training dataset preprocessing complete.")

print("Preprocessing validation dataset...")
val_dataset = val_dataset.map(preprocess_function, batched=True)
print("Validation dataset preprocessing complete.")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="output",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

print("Initializing Trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    print("Training completed.")
    print("Saving final model...")
    trainer.save_model("output/final_model")
    tokenizer.save_pretrained("output/final_model")
    print("Model and tokenizer saved to output/final_model")