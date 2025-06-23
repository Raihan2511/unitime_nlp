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


# preprocess.py
import re

def preprocess_data(offerings, reservations, preferences):
    dataset = []
    
    def extract_placeholders(text):
        """Extract placeholders like [PLACEHOLDER] from text"""
        return re.findall(r'\[([^\]]+)\]', text)
    
    def find_dataset_placeholders(items):
        """Find all unique placeholders used in a dataset"""
        placeholders = set()
        for item in items:
            input_text = item.get("input", "")
            placeholders.update(extract_placeholders(input_text))
        return placeholders
    
    def get_field_value(item, placeholder):
        """Get actual field value for a placeholder with flexible matching"""
        field_variations = [
            placeholder.lower(),
            placeholder.lower().replace('_', ''),
            placeholder.lower() + '_name',
            placeholder.lower() + '_id',
            placeholder.lower() + '_nbr'
        ]
        
        for field_name in field_variations:
            if field_name in item:
                return item[field_name]
        
        return f"[{placeholder}]"  # Keep placeholder if no match found
    
    def generate_prompt(item, dtype, dataset_placeholders):
        if dtype == "preferences":
            # Use detected placeholders or fallback to common ones
            dept = get_field_value(item, "DEPARTMENT") if "DEPARTMENT" in dataset_placeholders else item.get("department", "[DEPARTMENT]")
            instructor = get_field_value(item, "INSTRUCTOR") if "INSTRUCTOR" in dataset_placeholders else item.get("instructor_name", "[INSTRUCTOR_NAME]")
            building = get_field_value(item, "BUILDING") if "BUILDING" in dataset_placeholders else item.get("building", "[BUILDING]")
            time = get_field_value(item, "TIME") if "TIME" in dataset_placeholders else item.get("time", "morning")
            
            return (
                f"Instructor {instructor} from {dept} prefers {time} classes in building {building}.",
                item.get("output", "")
            )
        
        elif dtype == "reservations":
            subject = get_field_value(item, "SUBJECT") if "SUBJECT" in dataset_placeholders else item.get("subject", "[SUBJECT]")
            course = get_field_value(item, "COURSE_NBR") if "COURSE_NBR" in dataset_placeholders else item.get("course_id", "[COURSE_ID]")
            group = get_field_value(item, "GROUP") if "GROUP" in dataset_placeholders else item.get("group", "[GROUP]")
            major = get_field_value(item, "MAJOR") if "MAJOR" in dataset_placeholders else item.get("major", "[MAJOR]")
            limit = get_field_value(item, "LIMIT") if "LIMIT" in dataset_placeholders else item.get("limit", "[LIMIT]")
            
            return (
                f"There is a reservation for {subject} {course} for group {group} with major {major} and limit {limit}.",
                item.get("output", "")
            )
        
        elif dtype == "offerings":
            subject = get_field_value(item, "SUBJECT") if "SUBJECT" in dataset_placeholders else item.get("subject", "[SUBJECT]")
            course = get_field_value(item, "COURSE_NBR") if "COURSE_NBR" in dataset_placeholders else item.get("course_id", "[COURSE_ID]")
            dept = get_field_value(item, "DEPARTMENT") if "DEPARTMENT" in dataset_placeholders else item.get("department", "[DEPARTMENT]")
            instructor = get_field_value(item, "INSTRUCTOR") if "INSTRUCTOR" in dataset_placeholders else item.get("instructor_name", "[INSTRUCTOR_NAME]")
            
            return (
                f"Course {subject} {course} is offered by {dept} and taught by instructor {instructor}.",
                item.get("output", "")
            )
        
        return ("", "")
    
    # Process each dataset type
    datasets = [
        (preferences, "preferences"),
        (reservations, "reservations"), 
        (offerings, "offerings")
    ]
    
    for items, dtype in datasets:
        if items:  # Only process if dataset is not empty
            # Find placeholders used in this dataset
            dataset_placeholders = find_dataset_placeholders(items)
            
            # Generate prompts for each item
            for item in items:
                input_text, output_text = generate_prompt(item, dtype, dataset_placeholders)
                dataset.append({"input": input_text, "output": output_text})
    
    return dataset