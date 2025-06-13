# # src/data_preprocessing.py
# from datasets import Dataset
# import json

# def convert_to_flan_t5_format(examples):
#     inputs = []
#     targets = []
    
#     for example in examples:
#         # More specific prompts based on type
#         if 'type' in example:
#             type_specific_prompt = {
#                 'reservations': 'Create XML reservation',
#                 'course_offering': 'Generate XML course offering',
#                 'preferences': 'Create XML instructor preferences',
#                 'course_timetable': 'Generate XML course timetable'
#             }
#             prompt_prefix = type_specific_prompt.get(example['type'], 'Convert to XML')
#             input_text = f"{prompt_prefix}: {example['input']}"
#         else:
#             input_text = f"Convert this university scheduling request to XML format: {example['input']}"
        
#         target_text = example['output']  # Keep XML as-is
        
#         inputs.append(input_text)
#         targets.append(target_text)
    
#     return {"input": inputs, "target": targets}


# def prepare_datasets():
#     # Load generated data
#     with open("data/processed/train_dataset.json", "r") as f:
#         train_data = json.load(f)
    
#     with open("data/processed/val_dataset.json", "r") as f:
#         val_data = json.load(f)
    
#     # Convert to FLAN-T5 format
#     train_formatted = convert_to_flan_t5_format(train_data)
#     val_formatted = convert_to_flan_t5_format(val_data)
    
#     # Create Hugging Face datasets
#     train_dataset = Dataset.from_dict(train_formatted)
#     val_dataset = Dataset.from_dict(val_formatted)
    
#     return train_dataset, val_dataset

# src/data_preprocessing.py
from datasets import Dataset
import json
import xml.etree.ElementTree as ET
import re

def convert_to_flan_t5_format(examples):
    """Convert dataset to proper format for T5 training - optimized for your XML structure"""
    inputs = []
    targets = []
    types = []
    
    for example in examples:
        # Use exact prompts that match your training data patterns
        if 'type' in example:
            type_specific_prompt = {
                'reservations': 'Generate XML reservation data',
                'course_offering': 'Create XML course offering', 
                'preferences': 'Generate XML instructor preferences',
                'course_timetable': 'Create XML course timetable'
            }
            prompt_prefix = type_specific_prompt.get(example['type'], 'Convert to XML')
            input_text = f"{prompt_prefix}: {example['input']}"
            types.append(example['type'])
        else:
            # Fallback for examples without type
            input_text = f"Convert this university scheduling request to XML format: {example['input']}"
            types.append('unknown')
        
        # Use 'output' field as target (matching your dataset structure)
        target_text = example['output']  # Keep XML as-is
        
        inputs.append(input_text)
        targets.append(target_text)
    
    return {
        "input": inputs, 
        "target": targets,
        "type": types
    }

def validate_xml_examples(examples):
    """Enhanced XML validation specifically for your dataset format"""
    valid_count = 0
    total_count = len(examples)
    errors = []
    
    for i, example in enumerate(examples):
        try:
            # Check if 'output' field exists (your dataset structure)
            xml_content = example.get('output', example.get('target', ''))
            if not xml_content:
                errors.append(f"No XML content at index {i}")
                continue
                
            # Parse XML
            root = ET.fromstring(xml_content)
            valid_count += 1
            
            # Additional validation for your specific XML structure
            if not validate_xml_structure_for_dataset(root, example.get('type', '')):
                print(f"Warning: XML structure may not match expected type at index {i}")
                
        except ET.ParseError as e:
            error_msg = f"Invalid XML at index {i}: {e}"
            errors.append(error_msg)
            print(error_msg)
            print(f"Input: {example.get('input', 'N/A')}")
            print(f"Output: {xml_content}")
            print("-" * 50)
        except Exception as e:
            error_msg = f"Unexpected error at index {i}: {e}"
            errors.append(error_msg)
            print(error_msg)
    
    print(f"XML Validation: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%) valid")
    
    if errors:
        print(f"Found {len(errors)} validation errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    return valid_count == total_count

def validate_xml_structure_for_dataset(root, xml_type):
    """Validate XML structure matches expected type from your dataset"""
    expected_structures = {
        'preferences': {
            'root': 'preferences',
            'required_attrs': ['term', 'year', 'campus'],
            'child_elements': ['instructor']
        },
        'course_offering': {
            'root': 'offerings', 
            'required_attrs': ['campus', 'year', 'term'],
            'child_elements': ['offering']
        },
        'reservations': {
            'root': 'reservations',
            'required_attrs': ['campus', 'year', 'term'], 
            'child_elements': ['reservation']
        },
        'course_timetable': {
            'root': 'timetable',
            'required_attrs': ['campus', 'year', 'term'],
            'child_elements': ['class']
        }
    }
    
    if xml_type not in expected_structures:
        return True  # Skip validation for unknown types
    
    expected = expected_structures[xml_type]
    
    # Check root element
    if root.tag != expected['root']:
        print(f"Warning: Expected root <{expected['root']}>, got <{root.tag}>")
        return False
    
    # Check required attributes
    for attr in expected['required_attrs']:
        if attr not in root.attrib:
            print(f"Warning: Missing required attribute '{attr}' in root element")
            return False
    
    # Check for expected child elements
    child_tags = [child.tag for child in root]
    for expected_child in expected['child_elements']:
        if expected_child not in child_tags:
            print(f"Warning: Missing expected child element <{expected_child}>")
            return False
    
    return True

def fix_common_xml_issues(xml_string):
    """Fix common XML formatting issues in your dataset"""
    # Remove any leading/trailing whitespace
    xml_string = xml_string.strip()
    
    # Ensure proper attribute quoting (your data uses double quotes)
    xml_string = re.sub(r'([a-zA-Z_]+)=([^"\s>]+)(?=\s|>)', r'\1="\2"', xml_string)
    
    # Fix self-closing tag formatting
    xml_string = re.sub(r'<([^/>]+)\s*/>', r'<\1/>', xml_string)
    
    # Clean up excessive whitespace
    xml_string = re.sub(r'>\s+<', '><', xml_string)
    xml_string = re.sub(r'\s+', ' ', xml_string)
    
    return xml_string

def prepare_datasets():
    """Prepare datasets with enhanced validation for your specific XML format"""
    try:
        # Load generated data
        print("Loading datasets...")
        with open("data/processed/train_dataset.json", "r") as f:
            train_data = json.load(f)
        
        with open("data/processed/val_dataset.json", "r") as f:
            val_data = json.load(f)
        
        print(f"Loaded {len(train_data)} training examples")
        print(f"Loaded {len(val_data)} validation examples")
        
        # Check dataset structure
        if train_data:
            print("\n📋 Dataset structure analysis:")
            sample = train_data[0]
            print(f"Available keys: {list(sample.keys())}")
            
            # Validate key structure matches your examples
            required_keys = ['input', 'output']
            optional_keys = ['type', 'id', 'metadata']
            
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                print(f"❌ Missing required keys: {missing_keys}")
                return None, None
            
            present_optional = [key for key in optional_keys if key in sample]
            if present_optional:
                print(f"✅ Optional keys present: {present_optional}")
            
            # Show data type distribution if available
            if 'type' in sample:
                type_counts = {}
                for example in train_data:
                    example_type = example.get('type', 'unknown')
                    type_counts[example_type] = type_counts.get(example_type, 0) + 1
                
                print(f"\n📊 Training data type distribution:")
                for xml_type, count in sorted(type_counts.items()):
                    percentage = (count / len(train_data)) * 100
                    print(f"  {xml_type}: {count} examples ({percentage:.1f}%)")
        
        # Validate XML quality
        print("\n🔍 Validating XML quality...")
        train_valid = validate_xml_examples(train_data)
        val_valid = validate_xml_examples(val_data)
        
        if not train_valid or not val_valid:
            print("\n⚠️  XML validation issues found. Attempting to fix...")
            
            # Fix XML issues
            train_data = fix_xml_dataset(train_data)
            val_data = fix_xml_dataset(val_data)
            
            # Re-validate
            print("Re-validating after fixes...")
            train_valid = validate_xml_examples(train_data)
            val_valid = validate_xml_examples(val_data)
        
        if train_valid and val_valid:
            print("✅ All XML examples are valid!")
        else:
            print("❌ Some XML validation issues remain")
        
        # Convert to Hugging Face datasets format
        print("\n🔄 Converting to training format...")
        
        # Apply conversion function
        train_formatted = convert_to_flan_t5_format(train_data)
        val_formatted = convert_to_flan_t5_format(val_data)
        
        # Create Dataset objects
        train_dataset = Dataset.from_dict(train_formatted)
        val_dataset = Dataset.from_dict(val_formatted)
        
        # Print dataset info
        print(f"\n📦 Dataset ready for training:")
        print(f"  Training examples: {len(train_dataset)}")
        print(f"  Validation examples: {len(val_dataset)}")
        print(f"  Features: {train_dataset.features}")
        
        # Show sample formatted example
        if len(train_dataset) > 0:
            print(f"\n📝 Sample formatted example:")
            sample = train_dataset[0]
            print(f"Input: {sample['input'][:100]}...")
            print(f"Target: {sample['target'][:100]}...")
            print(f"Type: {sample.get('type', 'N/A')}")
        
        return train_dataset, val_dataset
        
    except FileNotFoundError as e:
        print(f"❌ Dataset files not found: {e}")
        print("Make sure you've run the data generation script first")
        return None, None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None, None

def fix_xml_dataset(dataset):
    """Fix XML formatting issues in the entire dataset"""
    fixed_dataset = []
    fix_count = 0
    
    for example in dataset:
        try:
            original_xml = example.get('output', example.get('target', ''))
            if not original_xml:
                continue
                
            # Attempt to fix common issues
            fixed_xml = fix_common_xml_issues(original_xml)
            
            # Try to parse the fixed XML
            ET.fromstring(fixed_xml)
            
            # If successful, update the example
            example_copy = example.copy()
            if 'output' in example_copy:
                example_copy['output'] = fixed_xml
            elif 'target' in example_copy:
                example_copy['target'] = fixed_xml
            
            fixed_dataset.append(example_copy)
            
            if fixed_xml != original_xml:
                fix_count += 1
                
        except ET.ParseError:
            # If still can't parse, skip this example
            print(f"⚠️  Skipping unparseable XML example: {example.get('input', 'N/A')[:50]}...")
            continue
        except Exception as e:
            print(f"⚠️  Error fixing example: {e}")
            # Keep original if fix fails
            fixed_dataset.append(example)
    
    print(f"🔧 Fixed {fix_count} XML formatting issues")
    print(f"📊 Kept {len(fixed_dataset)}/{len(dataset)} examples after fixing")
    
    return fixed_dataset

def analyze_dataset_quality(dataset):
    """Analyze dataset quality metrics"""
    if not dataset:
        return
    
    print("\n📈 Dataset Quality Analysis:")
    
    # Length statistics
    input_lengths = [len(example['input']) for example in dataset]
    output_lengths = [len(example.get('output', example.get('target', ''))) for example in dataset]
    
    print(f"Input length - avg: {sum(input_lengths)/len(input_lengths):.1f}, "
          f"min: {min(input_lengths)}, max: {max(input_lengths)}")
    print(f"Output length - avg: {sum(output_lengths)/len(output_lengths):.1f}, "
          f"min: {min(output_lengths)}, max: {max(output_lengths)}")
    
    # XML complexity analysis
    xml_complexity = analyze_xml_complexity(dataset)
    print(f"Average XML elements per example: {xml_complexity['avg_elements']:.1f}")
    print(f"Average XML attributes per example: {xml_complexity['avg_attributes']:.1f}")
    
    return {
        'input_lengths': input_lengths,
        'output_lengths': output_lengths,
        'xml_complexity': xml_complexity
    }

def analyze_xml_complexity(dataset):
    """Analyze XML structural complexity"""
    total_elements = 0
    total_attributes = 0
    valid_count = 0
    
    for example in dataset:
        try:
            xml_content = example.get('output', example.get('target', ''))
            root = ET.fromstring(xml_content)
            
            # Count elements
            elements = len(list(root.iter()))
            total_elements += elements
            
            # Count attributes
            attributes = sum(len(elem.attrib) for elem in root.iter())
            total_attributes += attributes
            
            valid_count += 1
            
        except ET.ParseError:
            continue
    
    if valid_count == 0:
        return {'avg_elements': 0, 'avg_attributes': 0}
    
    return {
        'avg_elements': total_elements / valid_count,
        'avg_attributes': total_attributes / valid_count
    }

def save_processed_datasets(train_dataset, val_dataset, output_dir="data/processed"):
    """Save processed datasets in multiple formats"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as HuggingFace datasets
    train_dataset.save_to_disk(f"{output_dir}/train_dataset_hf")
    val_dataset.save_to_disk(f"{output_dir}/val_dataset_hf")
    
    # Save as JSON for inspection
    with open(f"{output_dir}/train_processed.json", "w") as f:
        json.dump(train_dataset.to_dict(), f, indent=2)
    
    with open(f"{output_dir}/val_processed.json", "w") as f:
        json.dump(val_dataset.to_dict(), f, indent=2)
    
    print(f"✅ Processed datasets saved to {output_dir}")

if __name__ == "__main__":
    print("🚀 Starting data preprocessing...")
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets()
    
    if train_dataset is not None and val_dataset is not None:
        # Analyze quality
        print("\n🔍 Analyzing training data quality...")
        train_quality = analyze_dataset_quality(train_dataset.to_dict())
        
        print("\n🔍 Analyzing validation data quality...")
        val_quality = analyze_dataset_quality(val_dataset.to_dict())
        
        # Save processed datasets
        save_processed_datasets(train_dataset, val_dataset)
        
        print("\n✅ Data preprocessing complete!")
        print(f"Ready for training with {len(train_dataset)} training examples")
        print(f"and {len(val_dataset)} validation examples")
    else:
        print("❌ Data preprocessing failed!")