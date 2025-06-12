# src/inference.py
from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import json

app = Flask(__name__)

class UniTimeNLPService:
    def __init__(self, model_path="models/flan_t5_unitime_xml"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        
    def process_request(self, natural_language_input):
        """Process natural language input and return structured output"""
        input_formatted = f"Convert this scheduling request to structured format: {natural_language_input}"
        
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
                temperature=0.3,
                do_sample=False  # Deterministic for production
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            structured_output = json.loads(prediction)
            return {
                "success": True,
                "input": natural_language_input,
                "output": structured_output
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": str(e),
                "raw_output": prediction
            }

# Initialize service
nlp_service = UniTimeNLPService()

@app.route('/process', methods=['POST'])
def process_nlp():
    data = request.json
    input_text = data.get('text', '')
    
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400
    
    result = nlp_service.process_request(input_text)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": "flan-t5-unitime"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)