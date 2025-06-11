# src/fine_tuning.py - FIXED VERSION WITH LEARNING RATE
import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import json
import os

class UniTimeFlanT5Trainer:
    def __init__(self, model_name="google/flan-t5-small"):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add special tokens if needed
        special_tokens = ["<instructor>", "<course>", "<room>", "<time>", "<preference>"]
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def preprocess_function(self, examples):
        """Tokenize the examples"""
        inputs = examples["input"]
        targets = examples["target"]
        
        model_inputs = self.tokenizer(
            inputs, 
            max_length=512, 
            truncation=True, 
            padding="max_length"
        )
        
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, 
                max_length=512, 
                truncation=True, 
                padding="max_length"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def fine_tune(self, train_dataset, val_dataset, output_dir="models/flan_t5_unitime"):
        """Fine-tune the model"""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            self.preprocess_function, 
            batched=True, 
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function, 
            batched=True, 
            remove_columns=val_dataset.column_names
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        
        # Training arguments - FIXED: Added learning_rate and other missing parameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=5e-4,
            weight_decay=0.01,
            warmup_steps=100,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            evaluation_strategy="steps",  # <-- FIXED
            eval_steps=200,
            save_steps=400,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
            report_to=["tensorboard"],  # <-- FIXED
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            gradient_accumulation_steps=2,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            lr_scheduler_type="linear",
)

        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        try:
            trainer.train()
            
            # Save model explicitly
            print("Saving model...")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Verify files are saved
            required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
            for file in required_files:
                file_path = os.path.join(output_dir, file)
                if os.path.exists(file_path):
                    print(f"✅ {file} saved successfully")
                else:
                    print(f"❌ {file} not found!")
            
            print(f"Model saved to {output_dir}")
            
            # Create metadata file
            metadata = {
                "model_name": self.model_name,
                "training_examples": len(train_dataset),
                "validation_examples": len(val_dataset),
                "epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "status": "completed"
            }
            
            with open(f"{output_dir}/model_info.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            return trainer
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            # Save error info
            error_info = {
                "status": "failed",
                "error": str(e),
                "model_name": self.model_name
            }
            with open(f"{output_dir}/error_log.json", "w") as f:
                json.dump(error_info, f, indent=2)
            raise

# Training script
if __name__ == "__main__":
    from data_preprocessing import prepare_datasets
    
    print("Preparing datasets...")
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets()
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = UniTimeFlanT5Trainer("google/flan-t5-small")
    
    # Fine-tune
    try:
        trained_model = trainer.fine_tune(train_dataset, val_dataset)
        print("Fine-tuning completed successfully!")
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        print("Please check the error_log.json file for details.")