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


#preprocess.py
def preprocess_data(offerings, reservations, preferences):
    dataset = []
    for offer, reservation, pref in zip(offerings, reservations, preferences):
        try:
            input_text = (
                f"Department {pref['department']} requires {pref['building']}, "
                f"instructor {pref['instructor_first']} needs morning classes, "
                f"{offer['subject']} {offer['course_id']} lab likes computer facilities, "
                f"and class sections need room assignments."
            )

            xml_output = f"""
<preferences term=\"Fall\" year=\"2010\" campus=\"woebegon\">
  <department code=\"{pref['department']}\">
    <buildingPref building=\"{pref['building']}\" level=\"R\"/>
  </department>
  <instructor firstName=\"{pref['instructor_first']}\" lastName=\"{pref['instructor_last']}\" department=\"{pref['department']}\">
    <timePref level=\"R\">
      <pref days=\"MWF\" start=\"0800\" stop=\"1200\" level=\"R\"/>
    </timePref>
  </instructor>
  <subpart subject=\"{offer['subject']}\" course=\"{offer['course_id']}\" type=\"Lab\">
    <groupPref group=\"{pref['group']}\" level=\"1\"/>
  </subpart>
  <class subject=\"{offer['subject']}\" course=\"{offer['course_id']}\" type=\"{offer['class_type']}\">
    <roomPref building=\"{pref['building']}\" room=\"{offer['room_id']}\" level=\"1\"/>
  </class>
</preferences>"""

            dataset.append({"input": input_text, "xml": xml_output.strip()})
        except KeyError as e:
            print(f"Missing key in one of the records: {e}")

    return dataset
