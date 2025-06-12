import json
import torch
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import accuracy_score, classification_report 
# # type: ignore
import xml.etree.ElementTree as ET

class UniTimeNLPEvaluator:
    def __init__(self, model_path="models/flan_t5_unitime_xml"):
        self.model_path = model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        required_files = ["config.json", "tokenizer_config.json"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]

        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")

        self.tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

    def predict(self, input_text):
        input_formatted = f"Convert this scheduling request to structured format: {input_text}"
        inputs = self.tokenizer(input_formatted, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                temperature=0.7,  # Adjusted for better diversity
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def xml_equal(self, xml1, xml2):
        try:
            tree1 = ET.fromstring(xml1.strip())
            tree2 = ET.fromstring(xml2.strip())
            return ET.tostring(tree1) == ET.tostring(tree2)
        except Exception:
            return False

    def evaluate_xml_outputs(self, test_examples):
        correct = 0
        total = len(test_examples)
        mismatches = []

        for i, example in enumerate(test_examples):
            if i % 20 == 0:
                print(f"Evaluating: {i}/{total}")

            prediction = self.predict(example["input"])
            gold = example["output"]

            if self.xml_equal(prediction, gold):
                correct += 1
            else:
                mismatches.append({
                    "input": example["input"],
                    "predicted": prediction,
                    "expected": gold
                })

        accuracy = correct / total
        return {
            "xml_accuracy": accuracy,
            "mismatches": mismatches
        }

    def run_full_evaluation(self, test_file="data/processed/val_dataset.json"):
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        with open(test_file, "r") as f:
            test_examples = json.load(f)

        print("Running XML-based evaluation...")
        xml_results = self.evaluate_xml_outputs(test_examples)
        print(f"\nXML Accuracy: {xml_results['xml_accuracy']:.3f}")

        sample_size = min(3, len(test_examples))
        for i in range(sample_size):
            pred = self.predict(test_examples[i]["input"])
            print(f"\nSample {i+1}:")
            print(f"Input: {test_examples[i]['input']}")
            print(f"Predicted: {pred}")
            print(f"Expected: {test_examples[i]['output']}")

        results_file = f"{self.model_path}/evaluation_xml.json"
        with open(results_file, "w") as f:
            json.dump(xml_results, f, indent=2)

        print(f"\n✅ Results saved to: {results_file}")
        return xml_results

if __name__ == "__main__":
    try:
        evaluator = UniTimeNLPEvaluator()
        results = evaluator.run_full_evaluation()
        print("\n🎉 Evaluation completed successfully!")
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
