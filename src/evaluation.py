def evaluate_model(model_checkpoint, test_file_path):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from tqdm import tqdm
    import torch
    import json

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = load_dataset("json", data_files={"test": test_file_path})["test"]

    smooth_fn = SmoothingFunction().method2
    exact_match_count = 0
    bleu_scores = []

    results = []

    for item in tqdm(dataset):
        input_text = item["input"]
        expected_output = item["output"].strip()

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Exact match
        if prediction == expected_output:
            exact_match_count += 1

        # BLEU score
        bleu = sentence_bleu([expected_output.split()], prediction.split(), smoothing_function=smooth_fn)
        bleu_scores.append(bleu)

        results.append({
            "input": input_text,
            "expected": expected_output,
            "predicted": prediction,
            "bleu": bleu,
            "match": prediction == expected_output
        })

    exact_match_accuracy = exact_match_count / len(dataset)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    print(f"\n✅ Evaluation Complete:")
    print(f"🔹 Exact Match Accuracy: {exact_match_accuracy:.4f}")
    print(f"🔹 Average BLEU Score: {avg_bleu:.4f}")

    with open("data/eval_result/eval_predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    print("📄 Predictions saved to data/eval_predictions.json")
