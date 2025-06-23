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
def preprocess_data(offerings, reservations, preferences):
    dataset = []

    def generate_prompt(item, dtype):
        if dtype == "preferences":
            dept = item.get("department", "[DEPARTMENT]")
            instructor = item.get("instructor_name", "[INSTRUCTOR_NAME]")
            building = item.get("building", "[BUILDING]")
            time = item.get("time", "morning")
            return (
                f"Instructor {instructor} from {dept} prefers {time} classes in building {building}.",
                item.get("output", "")
            )

        elif dtype == "reservations":
            subject = item.get("subject", "[SUBJECT]")
            course = item.get("course_id", "[COURSE_ID]")
            group = item.get("group", "[GROUP]")
            major = item.get("major", "[MAJOR]")
            limit = item.get("limit", "[LIMIT]")
            return (
                f"There is a reservation for {subject} {course} for group {group} with major {major} and limit {limit}.",
                item.get("output", "")
            )

        elif dtype == "offerings":
            subject = item.get("subject", "[SUBJECT]")
            course = item.get("course_id", "[COURSE_ID]")
            dept = item.get("department", "[DEPARTMENT]")
            instructor = item.get("instructor_name", "[INSTRUCTOR_NAME]")
            return (
                f"Course {subject} {course} is offered by {dept} and taught by instructor {instructor}.",
                item.get("output", "")
            )

        return ("", "")

    for item in preferences:
        input_text, output_text = generate_prompt(item, "preferences")
        dataset.append({"input": input_text, "output": output_text})

    for item in reservations:
        input_text, output_text = generate_prompt(item, "reservations")
        dataset.append({"input": input_text, "output": output_text})

    for item in offerings:
        input_text, output_text = generate_prompt(item, "offerings")
        dataset.append({"input": input_text, "output": output_text})

    return dataset
