# def evaluate_model(model_checkpoint, test_file_path):
#     from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#     from datasets import load_dataset
#     from sklearn.metrics import accuracy_score
#     from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#     from tqdm import tqdm
#     import torch
#     import json

#     model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
#     tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()

#     dataset = load_dataset("json", data_files={"test": test_file_path})["test"]

#     smooth_fn = SmoothingFunction().method2
#     exact_match_count = 0
#     bleu_scores = []

#     results = []

#     for item in tqdm(dataset):
#         input_text = item["input"]
#         expected_output = item["output"].strip()

#         inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
#         outputs = model.generate(**inputs, max_new_tokens=512)
#         prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

#         # Exact match
#         if prediction == expected_output:
#             exact_match_count += 1

#         # BLEU score
#         bleu = sentence_bleu([expected_output.split()], prediction.split(), smoothing_function=smooth_fn)
#         bleu_scores.append(bleu)

#         results.append({
#             "input": input_text,
#             "expected": expected_output,
#             "predicted": prediction,
#             "bleu": bleu,
#             "match": prediction == expected_output
#         })

#     exact_match_accuracy = exact_match_count / len(dataset)
#     avg_bleu = sum(bleu_scores) / len(bleu_scores)

#     print(f"\n✅ Evaluation Complete:")
#     print(f"🔹 Exact Match Accuracy: {exact_match_accuracy:.4f}")
#     print(f"🔹 Average BLEU Score: {avg_bleu:.4f}")

#     with open("data/eval_result/eval_predictions.json", "w") as f:
#         json.dump(results, f, indent=2)
#     print("📄 Predictions saved to data/eval_predictions.json")


def evaluate_model(model_checkpoint, test_file_path, output_dir="data/eval_result"):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from datasets import load_dataset
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from tqdm import tqdm
    import torch
    import json
    import os
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    import re
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"test": test_file_path})["test"]
    
    smooth_fn = SmoothingFunction().method2
    exact_match_count = 0
    bleu_scores = []
    xml_structure_match_count = 0
    
    results = []
    
    def normalize_xml(xml_string):
        """Normalize XML for better comparison"""
        try:
            # Remove extra whitespace and normalize
            xml_string = re.sub(r'>\s+<', '><', xml_string.strip())
            # Parse and pretty print for consistent formatting
            root = ET.fromstring(xml_string)
            return ET.tostring(root, encoding='unicode')
        except ET.ParseError:
            return xml_string.strip()
    
    def check_xml_structure_match(expected, predicted):
        """Check if XML structures match (ignoring formatting differences)"""
        try:
            expected_root = ET.fromstring(expected)
            predicted_root = ET.fromstring(predicted)
            return ET.tostring(expected_root, encoding='unicode') == ET.tostring(predicted_root, encoding='unicode')
        except ET.ParseError:
            return False
    
    def is_valid_xml(xml_string):
        """Check if string is valid XML"""
        try:
            ET.fromstring(xml_string)
            return True
        except ET.ParseError:
            return False
    
    print("Starting evaluation...")
    for item in tqdm(dataset, desc="Evaluating"):
        input_text = item["input"]
        expected_output = item["output"].strip()
        
        # Tokenize input
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode prediction (remove input tokens for encoder-decoder models)
        if hasattr(model.config, 'is_encoder_decoder') and model.config.is_encoder_decoder:
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            # For decoder-only models, remove the input part
            input_length = inputs['input_ids'].shape[1]
            prediction = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        # Normalize both expected and predicted outputs for comparison
        normalized_expected = normalize_xml(expected_output)
        normalized_predicted = normalize_xml(prediction)
        
        # Exact match (with normalization)
        exact_match = normalized_predicted == normalized_expected
        if exact_match:
            exact_match_count += 1
        
        # XML structure match (more lenient)
        xml_structure_match = check_xml_structure_match(expected_output, prediction)
        if xml_structure_match:
            xml_structure_match_count += 1
        
        # BLEU score (token-based)
        bleu = sentence_bleu(
            [expected_output.split()], 
            prediction.split(), 
            smoothing_function=smooth_fn
        )
        bleu_scores.append(bleu)
        
        # Check if prediction is valid XML
        is_prediction_valid_xml = is_valid_xml(prediction)
        
        results.append({
            "input": input_text,
            "expected": expected_output,
            "predicted": prediction,
            "exact_match": exact_match,
            "xml_structure_match": xml_structure_match,
            "is_valid_xml": is_prediction_valid_xml,
            "bleu": bleu
        })
    
    # Calculate metrics
    total_samples = len(dataset)
    exact_match_accuracy = exact_match_count / total_samples
    xml_structure_accuracy = xml_structure_match_count / total_samples
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    valid_xml_count = sum(1 for r in results if r["is_valid_xml"])
    valid_xml_rate = valid_xml_count / total_samples
    
    # Print results
    print(f"\n✅ Evaluation Complete:")
    print(f"🔹 Total Samples: {total_samples}")
    print(f"🔹 Exact Match Accuracy: {exact_match_accuracy:.4f} ({exact_match_count}/{total_samples})")
    print(f"🔹 XML Structure Match Accuracy: {xml_structure_accuracy:.4f} ({xml_structure_match_count}/{total_samples})")
    print(f"🔹 Valid XML Rate: {valid_xml_rate:.4f} ({valid_xml_count}/{total_samples})")
    print(f"🔹 Average BLEU Score: {avg_bleu:.4f}")
    
    # Save detailed results
    results_file = os.path.join(output_dir, "eval_predictions.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary metrics
    summary = {
        "model_checkpoint": model_checkpoint,
        "total_samples": total_samples,
        "exact_match_accuracy": exact_match_accuracy,
        "xml_structure_accuracy": xml_structure_accuracy,
        "valid_xml_rate": valid_xml_rate,
        "average_bleu_score": avg_bleu,
        "exact_matches": exact_match_count,
        "xml_structure_matches": xml_structure_match_count,
        "valid_xml_predictions": valid_xml_count
    }
    
    summary_file = os.path.join(output_dir, "eval_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Detailed results saved to {results_file}")
    print(f"📊 Summary metrics saved to {summary_file}")
    
    return summary

# Example usage:
# results = evaluate_model("your-model-checkpoint", "path/to/test.json")