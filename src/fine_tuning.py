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

import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
import re
from typing import List, Dict, Tuple
import numpy as np
from data_preprocessing import prepare_datasets

class ImprovedXMLT5Trainer:
    def __init__(self, model_name="google/flan-t5-large"):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Enhanced XML tokens based on your actual dataset structure
        xml_tokens = [
            # Root elements (exact matches from your data)
            "<preferences>", "</preferences>", "<reservations>", "</reservations>",
            "<offerings>", "</offerings>", "<timetable>", "</timetable>",
            
            # Container elements
            "<instructor>", "</instructor>", "<department>", "</department>",
            "<student>", "</student>", "<reservation>", "</reservation>",
            "<offering>", "</offering>", "<class>", "</class>", "<course>", "</course>",
            "<time>", "</time>", "<room>", "</room>",
            
            # Preference elements
            "<timePref>", "</timePref>", "<pref>", "</pref>",
            "<coursePref>", "</coursePref>", "<roomPref>", "</roomPref>",
            "<distributionPref>", "</distributionPref>", "<subpart>", "</subpart>",
            
            # Self-closing tags (from your dataset)
            "<pref/>", "<coursePref/>", "<roomPref/>", "<student/>", "<time/>", "<room/>",
            
            # Specific attribute patterns from your dataset
            'term="Fall"', 'year="2024"', 'campus="MAIN"',
            'level="R"', 'level="P"', 'level="2"', 'level="-1"', 'level="-2"',
            'type="individual"', 'type="curriculum"', 'type="LEC"', 'type="LAB"', 'type="SEM"',
            'day="M"', 'day="T"', 'day="W"', 'day="R"', 'day="F"',
            'days="MWF"', 'days="TR"', 'days="WF"', 'days="MW"', 'days="R"',
            'offered="true"', 'lead="true"',
            
            # Building codes from your examples
            'building="EDUC"', 'building="SCI"', 'building="A"',
            'subject="PHYS"', 'subject="ENG"', 'subject="HIST"',
            
            # Common XML structure tokens
            'firstName=', 'lastName=', 'fname=', 'lname=',
            'startTime=', 'endTime=', 'roomNbr=', 'courseNbr=',
            'externalId=', 'expire=', 'limit=', 'suffix=',
            
            # Special tokens for structure
            "<XML_START>", "<XML_END>", "<ATTR_START>", "<ATTR_END>"
        ]
        
        # Add tokens to tokenizer
        num_added = self.tokenizer.add_tokens(xml_tokens)
        print(f"Added {num_added} XML-specific tokens to tokenizer")
        
        # Resize model embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))

    def classify_input_intent(self, input_text: str) -> str:
        """Enhanced intent classification based on your specific dataset patterns"""
        input_lower = input_text.lower()
        
        # More precise classification based on your actual examples
        if any(phrase in input_lower for phrase in ['reserve', 'seats', 'limit', 'until', 'student with id']):
            return 'reservations'
        elif any(phrase in input_lower for phrase in ['create course offering', 'offering', 'with limit', 'students']):
            return 'course_offering'
        elif any(phrase in input_lower for phrase in ['schedule', 'class id', 'timetable']):
            return 'course_timetable'
        elif any(phrase in input_lower for phrase in ['prefer', 'like', 'dislike', 'avoid', 'unavailable', 'strongly', 'instructor']):
            return 'preferences'
        else:
            # Enhanced fallback logic
            if 'class id' in input_lower or 'schedule' in input_lower:
                return 'course_timetable'
            elif any(title in input_lower for title in ['dr.', 'prof.', 'professor']) and 'offering' not in input_lower:
                if 'avoid' in input_lower or 'prefer' in input_lower:
                    return 'preferences'
                elif 'schedule' in input_lower:
                    return 'course_timetable'
                else:
                    return 'preferences'
            elif 'course' in input_lower and 'offering' in input_lower:
                return 'course_offering'
            else:
                return 'preferences'  # Default

    def create_structured_prompt(self, input_text: str, xml_type: str = None) -> str:
        """Create more structured prompts that match your dataset examples"""
        if xml_type:
            # Use exact prompts that match your training data structure
            type_prompts = {
                'reservations': 'Generate XML reservation data',
                'course_offering': 'Create XML course offering',
                'preferences': 'Generate XML instructor preferences',
                'course_timetable': 'Create XML course timetable'
            }
            prompt_prefix = type_prompts.get(xml_type, 'Convert to XML')
        else:
            # Classify intent if type not provided
            intent = self.classify_input_intent(input_text)
            type_prompts = {
                'reservations': 'Generate XML reservation data',
                'course_offering': 'Create XML course offering',
                'preferences': 'Generate XML instructor preferences',
                'course_timetable': 'Create XML course timetable'
            }
            prompt_prefix = type_prompts.get(intent, 'Convert to XML')
        
        return f"{prompt_prefix}: {input_text}"

    def preprocess_function(self, examples):
        """Fixed preprocessing function to match your exact dataset structure"""
        inputs = examples["input"]
        targets = examples["target"]
        
        # Create structured prompts - check if 'type' column exists
        enhanced_inputs = []
        if "type" in examples:
            types = examples["type"]
            for inp, xml_type in zip(inputs, types):
                enhanced_input = self.create_structured_prompt(inp, xml_type)
                enhanced_inputs.append(enhanced_input)
        else:
            for inp in inputs:
                enhanced_input = self.create_structured_prompt(inp)
                enhanced_inputs.append(enhanced_input)
        
        # Tokenize inputs with appropriate length for your data
        model_inputs = self.tokenizer(
            enhanced_inputs, 
            max_length=256,  # Sufficient for your input examples
            truncation=True, 
            padding="max_length",
            return_tensors="pt"
        )
        
        # Validate and clean targets based on your XML structure
        cleaned_targets = []
        for target in targets:
            # Validate XML structure
            try:
                ET.fromstring(target)
                cleaned_targets.append(target)
            except ET.ParseError as e:
                print(f"XML Parse Error: {e}")
                print(f"Problematic XML: {target}")
                # Try to fix common issues specific to your data format
                fixed_target = self.fix_xml_structure(target)
                try:
                    ET.fromstring(fixed_target)
                    cleaned_targets.append(fixed_target)
                    print(f"Fixed XML: {fixed_target}")
                except:
                    print(f"Could not fix XML, using original: {target}")
                    cleaned_targets.append(target)  # Use original even if invalid
        
        # Tokenize targets with longer length for complex XML
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                cleaned_targets, 
                max_length=768,  # Increased for complex XML structures like your examples
                truncation=True, 
                padding="max_length",
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def fix_xml_structure(self, xml_string: str) -> str:
        """Fix common XML structure issues specific to your dataset format"""
        # Remove any leading/trailing whitespace
        xml_string = xml_string.strip()
        
        # Ensure proper quotes around attributes (your data uses double quotes)
        xml_string = re.sub(r'([a-zA-Z_]+)=([^"\s>]+)(?=\s|>)', r'\1="\2"', xml_string)
        
        # Fix missing closing brackets
        xml_string = re.sub(r'<([^/>]+)(?<!/)$', r'<\1>', xml_string)
        
        # Fix self-closing tags format (ensure proper spacing)
        xml_string = re.sub(r'<([^/>]+)/\s*>', r'<\1/>', xml_string)
        
        # Fix attribute spacing issues
        xml_string = re.sub(r'\s+', ' ', xml_string)
        xml_string = re.sub(r'>\s+<', '><', xml_string)
        
        return xml_string

    def validate_xml(self, xml_string: str) -> bool:
        """Validate XML structure with better error handling"""
        try:
            if not xml_string.strip():
                return False
            # Remove any leading non-XML content
            if '<' in xml_string:
                xml_start = xml_string.find('<')
                xml_string = xml_string[xml_start:]
            ET.fromstring(xml_string)
            return True
        except ET.ParseError as e:
            print(f"XML validation error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected validation error: {e}")
            return False

    def fine_tune(self, train_dataset, val_dataset, output_dir="models/flan_t5_unitime_xml"):
        """Fine-tune with improved training strategy for your XML format"""
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        
        print(f"Output directory: {os.path.abspath(output_dir)}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Print dataset structure for debugging
        print("Dataset columns:", train_dataset.column_names)
        print("Sample from train dataset:", train_dataset[0])
        
        # Tokenize datasets
        print("Tokenizing datasets with improved preprocessing...")
        train_dataset_processed = train_dataset.map(
            self.preprocess_function, 
            batched=True, 
            remove_columns=train_dataset.column_names
        )
        
        val_dataset_processed = val_dataset.map(
            self.preprocess_function, 
            batched=True, 
            remove_columns=val_dataset.column_names
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Optimized training arguments for your XML generation task
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=15,  # More epochs for better XML structure learning
            per_device_train_batch_size=1,  # Smaller batch for complex XML
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,  # Compensate for very small batch size
            learning_rate=2e-5,  # Lower learning rate for better stability
            weight_decay=0.01,
            warmup_ratio=0.1,
            
            # Logging and evaluation
            logging_dir=f"{output_dir}/logs",
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=5,
            
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Optimization
            fp16=True,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            prediction_loss_only=False,
            
            # Regularization
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            
            # Reporting
            report_to=["tensorboard"],
            
            # Early stopping patience
            ignore_data_skip=True,
        )

        # Add early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=5,  # More patience for complex XML learning
            early_stopping_threshold=0.001
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset_processed,
            eval_dataset=val_dataset_processed,
            data_collator=data_collator,
            callbacks=[early_stopping]
        )
        
        print("Starting XML-focused training for your dataset format...")
        try:
            trainer.train()
            
            # Save model
            print("Saving model...")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Test with your exact dataset examples
            print("\n🧪 Testing inference with your dataset examples:")
            test_cases = [
                ("Instructor Prof. Johnson strongly avoids room EDUC201", "preferences"),
                ("Create course offering PHYS402 PHYS Course with Dr. Jones on Wednesday Friday from 11:00 AM to 12:00 PM in room SCI102 with limit 40 students", "course_offering"),
                ("Reserve 10 seats in ENG402 for student with ID STU73237 until 2024-12-01", "reservations"),
                ("Schedule HIST301 HIST Course with Prof. Martinez on Thursday from 6:00 PM to 7:00 PM in room A302, class ID 39134", "course_timetable")
            ]
            
            valid_count = 0
            for i, (test_input, expected_type) in enumerate(test_cases, 1):
                result = self.test_inference(test_input, expected_type)
                is_valid = self.validate_xml(result)
                
                print(f"\nTest {i} ({expected_type}):")
                print(f"Input: {test_input}")
                print(f"Output: {result}")
                print(f"Valid XML: {'✅' if is_valid else '❌'}")
                
                if is_valid:
                    valid_count += 1
                    # Check if it matches expected structure
                    try:
                        root = ET.fromstring(result)
                        print(f"Root element: <{root.tag}> ✅")
                        print(f"Root attributes: {root.attrib}")
                    except Exception as e:
                        print(f"Could not parse structure: {e}")
                else:
                    # Try to fix and validate again
                    try:
                        fixed_result = self.fix_xml_structure(result)
                        is_fixed_valid = self.validate_xml(fixed_result)
                        print(f"Fixed: {fixed_result}")
                        print(f"Fixed Valid: {'✅' if is_fixed_valid else '❌'}")
                        if is_fixed_valid:
                            valid_count += 1
                    except Exception as e:
                        print(f"Could not fix XML: {e}")
            
            print(f"\n📊 Overall XML Validity: {valid_count}/{len(test_cases)} ({valid_count/len(test_cases)*100:.1f}%)")
            
            return trainer
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_inference(self, input_text: str, xml_type: str = None) -> str:
        """Test inference with parameters optimized for your XML structure"""
        prompt = self.create_structured_prompt(input_text, xml_type)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=256, 
            truncation=True,
            padding=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=768,  # Increased for complex XML structures
                min_length=30,   # Ensure sufficient XML content
                num_beams=8,     # More beams for better structure
                do_sample=False, # Deterministic for XML structure
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                repetition_penalty=1.1,  # Lower penalty to allow proper XML repetition
                length_penalty=1.0,
                no_repeat_ngram_size=2,  # Allow some repetition for XML structure
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                temperature=0.1,  # Low temperature for consistent structure
                top_p=0.9
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to ensure proper XML structure matching your format
        result = self.post_process_xml(result)
        
        return result
    
    def post_process_xml(self, xml_string: str) -> str:
        """Post-process generated XML to match your dataset format exactly"""
        # Remove any non-XML prefix/suffix
        if '<' in xml_string:
            xml_start = xml_string.find('<')
            xml_string = xml_string[xml_start:]
        
        # Find the end of the XML (last complete closing tag)
        if xml_string.count('<') > 0:
            # Find last closing tag position
            last_close_tag = xml_string.rfind('</')
            if last_close_tag != -1:
                # Find the end of this closing tag
                tag_end = xml_string.find('>', last_close_tag)
                if tag_end != -1:
                    xml_string = xml_string[:tag_end + 1]
        
        # Fix attribute quoting to match your format (double quotes)
        xml_string = re.sub(r'([a-zA-Z_]+)=([^"\s>]+)(?=\s|>)', r'\1="\2"', xml_string)
        
        # Ensure proper attribute formatting
        xml_string = re.sub(r'=([^"]\w+)', r'="\1"', xml_string)
        
        # Clean up whitespace but preserve structure
        xml_string = re.sub(r'>\s+<', '><', xml_string)
        xml_string = re.sub(r'\s+', ' ', xml_string)
        
        # Ensure self-closing tags are properly formatted
        xml_string = re.sub(r'<([^/>]+)\s*/>', r'<\1/>', xml_string)
        
        return xml_string.strip()

def main():
    print("🚀 Starting XML-focused fine-tuning for your dataset...")

    try:
        print("🔄 Preparing datasets...")
        train_dataset, val_dataset = prepare_datasets()

        print(f"✅ Train dataset size: {len(train_dataset)}")
        print(f"✅ Validation dataset size: {len(val_dataset)}")
        
        # Show a sample from the dataset
        print("\n📦 Sample training example:")
        print("Dataset columns:", train_dataset.column_names)
        sample = train_dataset[0]
        print(f"Input: {sample['input']}")
        print(f"Target: {sample['target']}")
        if 'type' in sample:
            print(f"Type: {sample['type']}")

        # Initialize trainer with the specified model
        print("\n🏗️ Initializing trainer...")
        trainer = ImprovedXMLT5Trainer("google/flan-t5-large")

        # Fine-tune the model
        print("\n🏋️ Fine-tuning in progress...")
        trained_model = trainer.fine_tune(train_dataset, val_dataset)

        print("✅ XML-focused fine-tuning completed successfully!")
        
        # Test with your exact examples
        print("\n🧪 Final testing with your exact dataset format...")
        your_examples = [
            ("Instructor Prof. Johnson strongly avoids room EDUC201", "preferences"),
            ("Create course offering PHYS402 PHYS Course with Dr. Jones on Wednesday Friday from 11:00 AM to 12:00 PM in room SCI102 with limit 40 students", "course_offering"),
            ("Reserve 10 seats in ENG402 for student with ID STU73237 until 2024-12-01", "reservations"),
            ("Schedule HIST301 HIST Course with Prof. Martinez on Thursday from 6:00 PM to 7:00 PM in room A302, class ID 39134", "course_timetable")
        ]
        
        valid_results = 0
        for i, (test_input, expected_type) in enumerate(your_examples, 1):
            print(f"\n--- Test {i} ({expected_type}) ---")
            print(f"Input: {test_input}")
            
            result = trainer.test_inference(test_input, expected_type)
            is_valid = trainer.validate_xml(result)
            
            print(f"Generated XML: {result}")
            print(f"Valid XML: {'✅' if is_valid else '❌'}")
            
            if is_valid:
                valid_results += 1
                # Check structure matches expected type
                try:
                    root = ET.fromstring(result)
                    expected_roots = {
                        'preferences': 'preferences',
                        'course_offering': 'offerings',
                        'reservations': 'reservations', 
                        'course_timetable': 'timetable'
                    }
                    expected_root = expected_roots.get(expected_type, '')
                    if root.tag == expected_root:
                        print(f"Correct root element: <{root.tag}> ✅")
                    else:
                        print(f"Wrong root element: got <{root.tag}>, expected <{expected_root}> ❌")
                except Exception as e:
                    print(f"Could not analyze structure: {e}")
            else:
                print("❌ Generated invalid XML")
        
        print(f"\n📊 Final Test Results: {valid_results}/{len(your_examples)} ({valid_results/len(your_examples)*100:.1f}%) valid XML generated")

    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()