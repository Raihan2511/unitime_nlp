import json
import re
from datetime import datetime
import os
import random

def parse_nlp_to_xml(nlp_input):
    """Parse natural language input to XML format following the DTD"""

    subject_match = re.search(r'(\w+)\s+(\d+)', nlp_input)
    subject = subject_match.group(1) if subject_match else "SUBJ"
    course_nbr = subject_match.group(2) if subject_match else "101"

    section_match = re.search(r'section\s+(\d+)', nlp_input, re.IGNORECASE)
    suffix = section_match.group(1) if section_match else "1"

    type_match = re.search(r'(Lec|Lab|Rec|Sem|Course Ind)', nlp_input, re.IGNORECASE)
    class_type = type_match.group(1).replace(' ', '') if type_match else "Lec"

    days_map = {'Monday': 'M', 'Tuesday': 'T', 'Wednesday': 'W', 'Thursday': 'Th', 'Friday': 'F'}
    days = ""
    for day, abbr in days_map.items():
        if day in nlp_input:
            days += abbr
    if 'MWF' in nlp_input:
        days = "MWF"
    elif 'TTh' in nlp_input or 'TuTh' in nlp_input:
        days = "TTh"
    if not days:
        days = "MWF"

    time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', nlp_input)
    if time_match:
        hour = int(time_match.group(1))
        minute = time_match.group(2)
        ampm = time_match.group(3)
        if ampm == 'PM' and hour != 12:
            hour += 12
        elif ampm == 'AM' and hour == 12:
            hour = 0
        start_time = f"{hour:02d}{minute}"
    else:
        start_time = "0930"

    room_match = re.search(r'room\s+(\w+)\s+(\d+)', nlp_input, re.IGNORECASE)
    if room_match:
        building = room_match.group(1)
        room_nbr = room_match.group(2)
        room_xml = f'<room building="{building}" roomNbr="{room_nbr}"/>'
    else:
        room_xml = '<room building="MAIN" roomNbr="101"/>'

    time_pattern_match = re.search(r'time pattern\s+([^,\s]+)', nlp_input, re.IGNORECASE)
    time_pattern = f' timePattern="{time_pattern_match.group(1)}"' if time_pattern_match else ""

    date_pattern_match = re.search(r'(Full Term|Odd Wks|Even Wks)', nlp_input)
    date_pattern = f' datePattern="{date_pattern_match.group(1)}"' if date_pattern_match else ""

    xml_output = f'''<timetable campus="main" year="2024" term="Fall">
<class subject="{subject}" courseNbr="{course_nbr}" type="{class_type}" suffix="{suffix}">
<time days="{days}" startTime="{start_time}"{time_pattern}{date_pattern}/>
{room_xml}
</class>
</timetable>'''

    return xml_output

def generate_dataset():
    """Generate diverse NLP-to-XML dataset with multiple intents"""

    schedule_samples = [
        "Schedule ENG 402 Course Ind section 4 on Tuesday from 11:00 AM in room LIB 201",
        "Add MATH 101 Lec on MWF at 9:30 AM in EDUC 102",
        "Create CS 201 Lab section 2 Thursday 2PM to 4PM in room SCI 105"
    ]

    conversation_samples = [
        "Hello, how are you today?",
        "What's the weather like?",
        "Can you help me with something?",
        "Thank you for your assistance",
        "I need help understanding this concept"
    ]

    mixed_samples = [
        "Hi! Can you schedule BIOL 101 on Monday at 10AM in room LAB 201? Thanks!",
        "Good morning. I need to add CS 301 Sem section 1 on Tuesday Thursday 10:30 AM. Can you help?",
        "Hello there! Please create PHYS 102 Lab on Friday 1:30 PM in TECH 304. Appreciate it!"
    ]

    dataset = []

    for sample in schedule_samples:
        xml_output = parse_nlp_to_xml(sample)
        dataset.append({"input": sample, "output": xml_output})

    for sample in conversation_samples:
        dataset.append({"input": sample, "output": "I'm here to help! How can I assist you today?"})

    for sample in mixed_samples:
        xml_output = parse_nlp_to_xml(sample)
        response = f"Hello! I'll help you schedule that class.\n\n{xml_output}"
        dataset.append({"input": sample, "output": response})

    return dataset


def generate_timetable_dataset():
    dataset = generate_dataset()
    random.shuffle(dataset)

    total = len(dataset)
    train_split = int(0.7 * total)
    val_split = int(0.15 * total)

    train_data = dataset[:train_split]
    val_data = dataset[train_split:train_split + val_split]
    test_data = dataset[train_split + val_split:]

    os.makedirs("data/course_timetable", exist_ok=True)
    with open("data/course_timetable/train_timetable.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open("data/course_timetable/val_timetable.json", "w") as f:
        json.dump(val_data, f, indent=2)
    with open("data/course_timetable/test_timetable.json", "w") as f:
        json.dump(test_data, f, indent=2)

    for i, sample in enumerate(dataset, 1):
        print(f"Sample {i}")
        print(f"Input: {sample['input']}")
        print(f"Output: {sample['output']}")
        print("-" * 50)

# if __name__ == "__main__":
#     generate_timetable_dataset()
    print("✅ Course timetable dataset generated and saved to data/course_timetable/")