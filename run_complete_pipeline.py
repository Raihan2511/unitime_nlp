# run_complete_pipeline.py - FIXED VERSION
import subprocess
import os
import sys
import json

def check_file_exists(filepath):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"✅ {filepath} exists")
        return True
    else:
        print(f"❌ {filepath} missing")
        return False

def check_model_files(model_dir):
    """Check if all required model files exist"""
    required_files = [
        "config.json",
        "tokenizer_config.json", 
        "pytorch_model.bin",
        # "tokenizer.json"
    ]
    
    print(f"\n🔍 Checking model files in {model_dir}:")
    all_exist = True
    
    for file in required_files:
        filepath = os.path.join(model_dir, file)
        if not check_file_exists(filepath):
            all_exist = False
    
    return all_exist

def run_complete_pipeline():
    """Run the complete fine-tuning pipeline with error handling"""
    
    print("🚀 Starting UniTime NLP Fine-tuning Pipeline")
    
    # Step 1: Generate dataset
    print("\n📊 Step 1: Generating synthetic dataset...")
    try:
        result = os.system("python src/data_generation.py")
        if result != 0:
            print("❌ Dataset generation failed!")
            return False
        
        # Check if dataset files were created
        if not check_file_exists("data/processed/train_dataset.json"):
            print("❌ Training dataset not created!")
            return False
        if not check_file_exists("data/processed/val_dataset.json"):
            print("❌ Validation dataset not created!")
            return False
            
        print("✅ Dataset generation completed successfully!")
    except Exception as e:
        print(f"❌ Error in dataset generation: {e}")
        return False

    # Step 2: Fine-tune model
    print("\n🤖 Step 2: Fine-tuning FLAN-T5...")
    try:
        result = os.system("python src/fine_tuning.py")
        if result != 0:
            print("❌ Fine-tuning failed!")
            return False
        
        # Check if model was saved
        # model_dir = "models/flan_t5_unitime"
        model_dir = "models/flan_t5_unitime_xml"

        if not check_model_files(model_dir):
            print("❌ Model files not created properly!")
            return False
            
        print("✅ Fine-tuning completed successfully!")
    except Exception as e:
        print(f"❌ Error in fine-tuning: {e}")
        return False
    
    # Step 3: Evaluate model
    print("\n📈 Step 3: Evaluating model...")
    try:
        result = os.system("python src/evaluation.py")
        if result != 0:
            print("❌ Evaluation failed!")
            return False
        print("✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        return False
    
    # Step 4: Test inference
    print("\n🧪 Step 4: Testing inference...")
    try:
        test_inputs = [
            # "Professor Smith strongly prefers CS101 and CS102",
            "Create preference for Dr. Smith for courses CS101 and CS102 with strong preference",
            # "Dr. Jones dislikes teaching on Fridays",
            "Dr. Jones is unavailable to teach on Friday",
            # "Room 101 is available Monday through Wednesday"
            "Room 101 is available on Monday, Tuesday, and Wednesday"
        ]
        
        from src.evaluation import UniTimeNLPEvaluator
        evaluator = UniTimeNLPEvaluator()
        
        print("\n🔍 Sample predictions:")
        for i, test_input in enumerate(test_inputs):
            result = evaluator.predict(test_input)
            print(f"\nTest {i+1}:")
            print(f"Input: {test_input}")
            print(f"Output: {result}")
        
        print("✅ Inference testing completed!")
    except Exception as e:
        print(f"❌ Error in inference testing: {e}")
        return False
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 Pipeline completed successfully!")
    print("="*60)
    print("📁 Model location: models/flan_t5_unitime/")
    print("📊 Dataset location: data/processed/")
    print("📈 Evaluation results: models/flan_t5_unitime/evaluation_results.json")
    print("🌐 To start API server: python src/inference.py")
    print("="*60)
    
    return True

def main():
    """Main function with error handling"""
    try:
        success = run_complete_pipeline()
        if success:
            print("\n✅ All steps completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Pipeline failed. Check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()