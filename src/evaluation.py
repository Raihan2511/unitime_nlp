# src/evaluation.py - FIXED VERSION
import json
import torch
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import accuracy_score, classification_report
import re

class UniTimeNLPEvaluator:
    def __init__(self, model_path="models/flan_t5_unitime"):
        self.model_path = model_path
        
        # Check if model exists locally
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}")
            print("Please ensure the fine-tuning step completed successfully.")
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Check if required files exist
        required_files = ["config.json", "tokenizer_config.json"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        
        if missing_files:
            print(f"⚠️  Missing model files: {missing_files}")
            print("Model training may not have completed successfully.")
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
            self.model.eval()
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def predict(self, input_text):
        """Generate prediction for input text"""
        input_formatted = f"Convert this scheduling request to structured format: {input_text}"
        
        inputs = self.tokenizer(
            input_formatted, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            return json.loads(prediction)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON output", "raw_output": prediction}

    def evaluate_intent_classification(self, test_examples):
        """Evaluate intent classification accuracy"""
        correct = 0
        total = len(test_examples)
        predictions = []
        actuals = []
        
        print(f"Evaluating intent classification on {total} examples...")
        
        for i, example in enumerate(test_examples):
            if i % 50 == 0:
                print(f"Progress: {i}/{total}")
                
            prediction = self.predict(example["input"])
            
            if "intent" in prediction:
                predicted_intent = prediction["intent"]
                actual_intent = example["output"]["intent"]
                
                predictions.append(predicted_intent)
                actuals.append(actual_intent)
                
                if predicted_intent == actual_intent:
                    correct += 1
            else:
                predictions.append("UNKNOWN")
                actuals.append(example["output"]["intent"])
        
        accuracy = correct / total
        report = classification_report(actuals, predictions, zero_division=0)
        
        return {
            "intent_accuracy": accuracy,
            "classification_report": report,
            "predictions": predictions,
            "actuals": actuals
        }

    def evaluate_entity_extraction(self, test_examples):
        """Evaluate entity extraction"""
        entity_scores = {}
        
        print(f"Evaluating entity extraction...")
        
        for i, example in enumerate(test_examples):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(test_examples)}")
                
            prediction = self.predict(example["input"])
            actual_entities = example["output"]["entities"]
            
            if "entities" in prediction:
                predicted_entities = prediction["entities"]
                
                for entity_type in actual_entities:
                    if entity_type not in entity_scores:
                        entity_scores[entity_type] = {"correct": 0, "total": 0}
                    
                    entity_scores[entity_type]["total"] += 1
                    
                    if entity_type in predicted_entities:
                        if predicted_entities[entity_type] == actual_entities[entity_type]:
                            entity_scores[entity_type]["correct"] += 1
        
        # Calculate accuracy for each entity type
        for entity_type in entity_scores:
            total = entity_scores[entity_type]["total"]
            correct = entity_scores[entity_type]["correct"]
            entity_scores[entity_type]["accuracy"] = correct / total if total > 0 else 0
        
        return entity_scores

    def run_full_evaluation(self, test_file="data/processed/val_dataset.json"):
        """Run complete evaluation"""
        
        if not os.path.exists(test_file):
            print(f"⚠️  Test file not found: {test_file}")
            print("Please ensure the dataset generation step completed successfully.")
            return None
            
        with open(test_file, "r") as f:
            test_examples = json.load(f)
        
        print("Running evaluation...")
        
        # Intent classification
        intent_results = self.evaluate_intent_classification(test_examples)
        print(f"Intent Classification Accuracy: {intent_results['intent_accuracy']:.3f}")
        
        # Entity extraction
        entity_results = self.evaluate_entity_extraction(test_examples)
        print("\nEntity Extraction Results:")
        for entity_type, scores in entity_results.items():
            print(f"  {entity_type}: {scores['accuracy']:.3f}")
        
        # Sample predictions
        print("\nSample Predictions:")
        for i, example in enumerate(test_examples[:3]):  # Reduced to 3 examples
            prediction = self.predict(example["input"])
            print(f"\nExample {i+1}:")
            print(f"Input: {example['input']}")
            print(f"Predicted: {prediction}")
            print(f"Actual: {example['output']}")
        
        # Save results
        results = {
            "intent_results": intent_results,
            "entity_results": entity_results,
            "model_path": self.model_path
        }
        
        results_file = f"{self.model_path}/evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {results_file}")
        
        return results

# Run evaluation
if __name__ == "__main__":
    try:
        evaluator = UniTimeNLPEvaluator()
        results = evaluator.run_full_evaluation()
        
        if results:
            print("\n🎉 Evaluation completed successfully!")
        else:
            print("\n❌ Evaluation failed!")
            
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        print("\nPossible solutions:")
        print("1. Ensure fine-tuning completed successfully")
        print("2. Check if model files exist in models/flan_t5_unitime/")
        print("3. Run the fine-tuning step again")