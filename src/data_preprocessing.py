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
import json
import os
def preprocess_data(offerings, reservations, preferences,timetable):
    dataset = []

    def generate_prompt(item, dtype):
        if dtype == "preferences":
            instructor = item.get("INSTRUCTOR_NAME", "[INSTRUCTOR_NAME]")
            dept = item.get("DEPARTMENT", "[DEPARTMENT]")
            building = item.get("BUILDING", "[BUILDING]")
            time = f"from {item.get('START_TIME', '[START_TIME]')} to {item.get('END_TIME', '[END_TIME]')}"
            subject = item.get("SUBJECT", "[SUBJECT]")
            course = item.get("COURSE_ID", "[COURSE_ID]")
            room = item.get("ROOM_ID", "[ROOM_ID]")
            class_type = item.get("CLASS_TYPE", "[CLASS_TYPE]")
            return (
                f"Instructor {instructor} from department {dept} prefers {class_type} classes of {subject} {course} scheduled {time} in building {building}, room {room}.",
                item.get("output", "")
            )

        elif dtype == "reservations":
            subject = item.get("SUBJECT", "[SUBJECT]")
            course = item.get("COURSE_NBR", "[COURSE_NBR]")
            major = item.get("MAJOR", "[MAJOR]")
            limit = item.get("LIMIT", "[LIMIT]")
            campus = item.get("CAMPUS", "[CAMPUS]")
            term = item.get("TERM", "[TERM]")
            year = item.get("YEAR", "[YEAR]")
            return (
                f"A reservation is made on {subject} {course} for major {major} with limit {limit} at {campus} campus during {term} {year}.",
                item.get("output", "")
            )

        elif dtype == "offerings":
            subject = item.get("SUBJECT", "[SUBJECT]")
            course = item.get("COURSE_NBR", "[COURSE_NBR]")
            instructor = item.get("INSTRUCTOR_LNAME", "[INSTRUCTOR_LNAME]")
            building = item.get("BUILDING", "[BUILDING]")
            start = item.get("START_TIME", "[START_TIME]")
            end = item.get("END_TIME", "[END_TIME]")
            room = item.get("ROOM_NBR", "[ROOM_NBR]")
            return (
                f"Course offering for {subject} {course} taught by instructor {instructor} in building {building}, room {room}, from {start} to {end}.",
                item.get("output", "")
            )
        elif dtype == "timetable":
            subject = item.get("SUBJECT", "[SUBJECT]")
            course = item.get("COURSE_NBR", "[COURSE_NBR]")
            class_type = item.get("CLASS_TYPE", "[CLASS_TYPE]")
            days = item.get("DAYS", "[DAYS]")
            start_time = item.get("START_TIME", "[START_TIME]")
            end_time = item.get("END_TIME", "[END_TIME]")
            building = item.get("BUILDING", "[BUILDING]")
            room = item.get("ROOM_NBR", "[ROOM_NBR]")
            return (
                f"Timetable entry for {subject} {course} ({class_type}) on {days} from {start_time} to {end_time} in {building}, room {room}.",
                item.get("output", "")
            )

        return ("", item.get("output", ""))

    for item in preferences:
        input_text, output_text = generate_prompt(item, "preferences")
        dataset.append({"input": input_text, "output": output_text})

    for item in reservations:
        input_text, output_text = generate_prompt(item, "reservations")
        dataset.append({"input": input_text, "output": output_text})

    for item in offerings:
        input_text, output_text = generate_prompt(item, "offerings")
        dataset.append({"input": input_text, "output": output_text})
    for item in timetable:
        input_text, output_text = generate_prompt(item, "timetable")
        dataset.append({"input": input_text, "output": output_text})

    return dataset

def merge_test_data(offer_data, reserv_data, pref_data,time_data, output_path):
    import os
    import json

    merged = []
    merged.extend(offer_data)
    merged.extend(reserv_data)
    merged.extend(pref_data)
    merged.extend(time_data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)


    print(f"✅ Merged test data saved to {output_path}")

# # Paths
# merge_test_data(
#     "data/offerings_data/test_offer.json",
#     "data/reservation_data/test_reservation.json",
#     "data/preference_data/test_pref.json",
#     "data/processed/test.json"
# )

