import json
import re
from datetime import datetime
import os
import random

def parse_nlp_to_xml(nlp_input):
    """Parse natural language input to XML format following the DTD strictly"""

    # Extract subject and course number
    subject_match = re.search(r'(\w+)\s+(\d+)', nlp_input)
    subject = subject_match.group(1) if subject_match else "SUBJ"
    course_nbr = subject_match.group(2) if subject_match else "101"

    # Extract section suffix
    section_match = re.search(r'section\s+(\d+)', nlp_input, re.IGNORECASE)
    suffix = section_match.group(1) if section_match else "1"

    # Extract class type
    type_match = re.search(r'(Lec|Lecture|Lab|Laboratory|Rec|Recitation|Sem|Seminar|Ind|Independent)', nlp_input, re.IGNORECASE)
    if type_match:
        type_found = type_match.group(1).lower()
        if type_found in ['lec', 'lecture']:
            class_type = "Lec"
        elif type_found in ['lab', 'laboratory']:
            class_type = "Lab"
        elif type_found in ['rec', 'recitation']:
            class_type = "Rec"
        elif type_found in ['sem', 'seminar']:
            class_type = "Sem"
        elif type_found in ['ind', 'independent']:
            class_type = "Ind"
        else:
            class_type = "Lec"
    else:
        class_type = "Lec"

    # Extract days
    days_map = {'Monday': 'M', 'Tuesday': 'T', 'Wednesday': 'W', 'Thursday': 'Th', 'Friday': 'F', 'Saturday': 'S', 'Sunday': 'Su'}
    days = ""
    for day, abbr in days_map.items():
        if day.lower() in nlp_input.lower():
            days += abbr
    
    # Check for common patterns
    if 'MWF' in nlp_input.upper():
        days = "MWF"
    elif any(pattern in nlp_input.upper() for pattern in ['TTH', 'TUTH', 'TU TH', 'TUESDAY THURSDAY']):
        days = "TTh"
    elif 'MW' in nlp_input.upper() or 'MONDAY WEDNESDAY' in nlp_input.upper():
        days = "MW"
    
    if not days:
        days = "MWF"  # Default

    # Extract start time
    time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', nlp_input, re.IGNORECASE)
    if time_match:
        hour = int(time_match.group(1))
        minute = time_match.group(2)
        ampm = time_match.group(3).upper()
        if ampm == 'PM' and hour != 12:
            hour += 12
        elif ampm == 'AM' and hour == 12:
            hour = 0
        start_time = f"{hour:02d}{minute}"
    else:
        start_time = "0930"

    # Extract end time if provided
    end_time_match = re.search(r'to\s+(\d{1,2}):(\d{2})\s*(AM|PM)', nlp_input, re.IGNORECASE) or \
                    re.search(r'-\s*(\d{1,2}):(\d{2})\s*(AM|PM)', nlp_input, re.IGNORECASE)
    end_time = ""
    if end_time_match:
        hour = int(end_time_match.group(1))
        minute = end_time_match.group(2)
        ampm = end_time_match.group(3).upper()
        if ampm == 'PM' and hour != 12:
            hour += 12
        elif ampm == 'AM' and hour == 12:
            hour = 0
        end_time = f' endTime="{hour:02d}{minute}"'

    # Extract room information
    room_match = re.search(r'room\s+(\w+)\s+(\d+)', nlp_input, re.IGNORECASE) or \
                re.search(r'in\s+(\w+)\s+(\d+)', nlp_input, re.IGNORECASE)
    if room_match:
        building = room_match.group(1).upper()
        room_nbr = room_match.group(2)
        room_xml = f'<room building="{building}" roomNbr="{room_nbr}"/>'
    else:
        room_xml = '<room building="MAIN" roomNbr="101"/>'

    # Extract instructor information
    instructor_match = re.search(r'instructor\s+([A-Za-z]+)\s+([A-Za-z]+)', nlp_input, re.IGNORECASE) or \
                      re.search(r'prof(?:essor)?\s+([A-Za-z]+)\s+([A-Za-z]+)', nlp_input, re.IGNORECASE) or \
                      re.search(r'taught by\s+([A-Za-z]+)\s+([A-Za-z]+)', nlp_input, re.IGNORECASE)
    instructor_xml = ""
    if instructor_match:
        fname = instructor_match.group(1)
        lname = instructor_match.group(2)
        instructor_xml = f'<instructor fname="{fname}" lname="{lname}"/>'

    # Extract date pattern
    date_pattern_match = re.search(r'(Full Term|Odd Wks|Even Wks|First Half|Second Half)', nlp_input, re.IGNORECASE)
    date_pattern = f' datePattern="{date_pattern_match.group(1)}"' if date_pattern_match else ""

    # Extract time pattern
    time_pattern_match = re.search(r'time pattern\s+([^,\s]+)', nlp_input, re.IGNORECASE)
    time_pattern = f' timePattern="{time_pattern_match.group(1)}"' if time_pattern_match else ""

    # Extract schedule note
    note_match = re.search(r'note[:\s]+([^.]+)', nlp_input, re.IGNORECASE)
    schedule_note = f' scheduleNote="{note_match.group(1).strip()}"' if note_match else ""

    # Extract class limit
    limit_match = re.search(r'limit\s+(\d+)', nlp_input, re.IGNORECASE) or \
                 re.search(r'capacity\s+(\d+)', nlp_input, re.IGNORECASE)
    limit = f' limit="{limit_match.group(1)}"' if limit_match else ""

    # Generate XML following DTD structure
    xml_output = f'''<timetable campus="main" year="2024" term="Fall">
<class subject="{subject}" courseNbr="{course_nbr}" type="{class_type}" suffix="{suffix}"{schedule_note}{limit}>
<time days="{days}" startTime="{start_time}"{end_time}{time_pattern}{date_pattern}/>
{room_xml}
{instructor_xml}
</class>
</timetable>'''

    return xml_output

def generate_diverse_dataset():
    """Generate comprehensive diverse NLP-to-XML dataset with single intent"""

    # Core subjects for variation
    subjects = ["ENG", "MATH", "CS", "PHYS", "CHEM", "BIOL", "HIST", "PSYC", "ECON", "STAT", 
               "ART", "MUSC", "THEA", "NURS", "BUSI", "EDUC", "POLI", "SOCI", "PHIL", "FREN"]
    
    course_numbers = ["101", "102", "201", "202", "301", "302", "401", "402", "501", "502"]
    
    # Basic scheduling samples - expanded
    basic_samples = [
        "Schedule ENG 402 Independent section 4 on Tuesday from 11:00 AM in room LIB 201",
        "Add MATH 101 Lecture on MWF at 9:30 AM in EDUC 102",
        "Create CS 201 Lab section 2 Thursday 2:00 PM to 4:00 PM in room SCI 105",
        "Set up PHYS 301 Seminar section 1 Monday Wednesday Friday 10:15 AM",
        "Book HIST 250 Recitation on TTh at 1:30 PM in ARTS 204",
        "Arrange BIOL 101 Laboratory section 3 Friday 2:00 PM to 5:00 PM",
        "Plan CHEM 202 Lecture MWF 8:00 AM in SCI 301 limit 30",
        "Schedule PSYC 101 on Tuesday Thursday 11:00 AM to 12:15 PM"
    ]
    
    # Generate more basic variations
    for i in range(15):
        subj = random.choice(subjects)
        num = random.choice(course_numbers)
        day_patterns = ["MWF", "TTh", "MW", "Friday", "Monday", "Tuesday", "Wednesday", "Thursday"]
        times = ["8:00 AM", "9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM", "3:00 PM"]
        types = ["Lecture", "Lab", "Seminar", "Recitation"]
        
        basic_samples.append(f"Schedule {subj} {num} {random.choice(types)} on {random.choice(day_patterns)} at {random.choice(times)}")

    # With instructor information - expanded
    instructor_samples = [
        "Schedule MATH 205 Lecture with Professor Smith Johnson on MWF 9:00 AM in MATH 101",
        "Add CS 301 Lab taught by John Davis section 2 Thursday 3:00 PM in COMP 205",
        "Create ENG 102 Seminar with instructor Mary Wilson TTh 10:30 AM",
        "Set PHYS 401 Laboratory with Prof Robert Brown Friday 1:00 PM to 4:00 PM",
        "Book HIST 301 Lecture instructor Sarah Miller MW 2:00 PM in ARTS 150"
    ]
    
    # Generate more instructor variations
    first_names = ["John", "Mary", "Robert", "Sarah", "David", "Lisa", "Michael", "Jennifer", "James", "Amanda"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    instructor_titles = ["Professor", "Prof", "Dr", "instructor"]
    
    for i in range(20):
        subj = random.choice(subjects)
        num = random.choice(course_numbers)
        fname = random.choice(first_names)
        lname = random.choice(last_names)
        title = random.choice(instructor_titles)
        day = random.choice(["MWF", "TTh", "MW", "Friday"])
        time = random.choice(["9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM"])
        
        instructor_samples.append(f"Create {subj} {num} with {title} {fname} {lname} on {day} at {time}")

    # With special patterns and notes - expanded
    advanced_samples = [
        "Schedule BIOL 301 Lab section 1 Odd Wks Friday 9:00 AM in LAB 301 capacity 20",
        "Add CHEM 401 Lecture Full Term MWF 11:00 AM note: Requires lab component",
        "Create MATH 301 Seminar Even Wks Tuesday 4:00 PM limit 15",
        "Set PHYS 201 Laboratory First Half MW 2:00 PM to 4:00 PM",
        "Book CS 401 Independent Second Half by arrangement note: Senior capstone project"
    ]
    
    # Generate more advanced variations
    patterns = ["Full Term", "Odd Wks", "Even Wks", "First Half", "Second Half"]
    notes = ["Requires prerequisites", "Lab component required", "Senior project", "Capstone course", "Research intensive"]
    limits = ["15", "20", "25", "30", "12", "18"]
    
    for i in range(15):
        subj = random.choice(subjects)
        num = random.choice(course_numbers)
        pattern = random.choice(patterns)
        note = random.choice(notes)
        limit = random.choice(limits)
        day = random.choice(["MWF", "TTh", "Friday", "Tuesday"])
        time = random.choice(["9:00 AM", "11:00 AM", "2:00 PM", "3:00 PM"])
        
        advanced_samples.append(f"Add {subj} {num} {pattern} {day} {time} limit {limit} note: {note}")

    # Different time formats and patterns - expanded
    time_variant_samples = [
        "Schedule ENG 201 at 8:00 AM on Monday Wednesday Friday in ARTS 205",
        "Add MATH 301 from 1:00 PM to 2:15 PM Tuesday Thursday",
        "Create BIOL 401 Laboratory 9:00 AM - 12:00 PM Friday in SCI 401",
        "Set HIST 101 Lecture 3:30 PM MWF room ARTS 301",
        "Book PSYC 205 Seminar 7:00 PM Tuesday in PSYC 101"
    ]
    
    # Generate more time variations
    start_times = ["8:00 AM", "8:30 AM", "9:00 AM", "9:30 AM", "10:00 AM", "10:30 AM", "11:00 AM", "11:30 AM", 
                  "12:00 PM", "12:30 PM", "1:00 PM", "1:30 PM", "2:00 PM", "2:30 PM", "3:00 PM", "3:30 PM", "4:00 PM"]
    end_times = ["9:15 AM", "9:50 AM", "10:15 AM", "10:50 AM", "11:15 AM", "11:50 AM", "12:15 PM", "12:50 PM",
                "1:15 PM", "1:50 PM", "2:15 PM", "2:50 PM", "3:15 PM", "3:50 PM", "4:15 PM", "4:50 PM", "5:15 PM"]
    
    for i in range(20):
        subj = random.choice(subjects)
        num = random.choice(course_numbers)
        start = random.choice(start_times)
        end = random.choice(end_times)
        day = random.choice(["MWF", "TTh", "MW", "Tuesday", "Thursday", "Friday"])
        
        formats = [
            f"Schedule {subj} {num} at {start} on {day}",
            f"Add {subj} {num} from {start} to {end} {day}",
            f"Create {subj} {num} {start} - {end} on {day}",
            f"Set {subj} {num} {day} at {start}"
        ]
        time_variant_samples.append(random.choice(formats))

    # Various room formats - expanded
    room_variant_samples = [
        "Schedule CS 101 in room COMP 205 MWF 10:00 AM",
        "Add MATH 201 Laboratory in SCI 301 Thursday 2:00 PM",
        "Create ENG 301 Seminar room LIB 205 TTh 11:30 AM",
        "Set PHYS 101 in PHYS 401 Monday Wednesday 9:00 AM",
        "Book CHEM 301 Lab in room LAB 205 Friday 1:00 PM"
    ]
    
    # Generate more room variations
    buildings = ["SCI", "ARTS", "COMP", "LAB", "LIB", "MATH", "PHYS", "CHEM", "ENG", "BUSI", "EDUC", "PSYC"]
    room_numbers = ["101", "102", "103", "201", "202", "203", "301", "302", "303", "401", "402", "403"]
    
    for i in range(15):
        subj = random.choice(subjects)
        num = random.choice(course_numbers)
        building = random.choice(buildings)
        room_num = random.choice(room_numbers)
        day = random.choice(["MWF", "TTh", "MW", "Friday", "Tuesday"])
        time = random.choice(["9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM"])
        
        formats = [
            f"Schedule {subj} {num} in room {building} {room_num} {day} {time}",
            f"Add {subj} {num} in {building} {room_num} {day} at {time}",
            f"Create {subj} {num} room {building} {room_num} on {day} {time}",
            f"Book {subj} {num} {building} {room_num} {day} {time}"
        ]
        room_variant_samples.append(random.choice(formats))

    # Mixed complexity samples - expanded
    complex_samples = [
        "Schedule advanced MATH 401 Seminar section 2 with Professor Davis on Tuesday Thursday 2:30 PM to 3:45 PM in MATH 205 Full Term limit 12",
        "Create BIOL 301 Laboratory section 1 taught by Dr. Sarah Wilson Odd Wks Friday 9:00 AM to 12:00 PM in LAB 301 capacity 16 note: Microscopy required",
        "Add CS 401 Independent Study section 3 with instructor John Smith by arrangement note: Senior capstone project limit 5",
        "Set up ENG 205 Lecture with Prof Mary Johnson MWF 10:15 AM in ARTS 150 Second Half capacity 25",
        "Book PHYS 401 Advanced Laboratory TTh 1:00 PM to 4:00 PM in PHYS 301 taught by Robert Brown Even Wks limit 8"
    ]
    
    # Generate more complex combinations
    for i in range(15):
        subj = random.choice(subjects)
        num = random.choice(course_numbers)
        fname = random.choice(first_names)
        lname = random.choice(last_names)
        title = random.choice(instructor_titles)
        pattern = random.choice(patterns)
        note = random.choice(notes)
        limit = random.choice(limits)
        building = random.choice(buildings)
        room_num = random.choice(room_numbers)
        day = random.choice(["MWF", "TTh", "MW", "Friday"])
        start = random.choice(start_times)
        end = random.choice(end_times)
        section = random.choice(["1", "2", "3", "4"])
        
        complex_samples.append(f"Create {subj} {num} section {section} with {title} {fname} {lname} {pattern} {day} {start} to {end} in {building} {room_num} limit {limit} note: {note}")

    # Weekend and evening classes - expanded  
    extended_samples = [
        "Schedule BUSI 301 Saturday 9:00 AM to 12:00 PM in BUSI 201",
        "Add NURS 401 Laboratory Sunday 1:00 PM to 5:00 PM",
        "Create ART 201 Studio Monday 6:00 PM to 9:00 PM in ART 301",
        "Set MUSC 101 Ensemble Thursday 7:30 PM in MUSC 205",
        "Book THEA 301 Workshop Friday 5:00 PM to 8:00 PM"
    ]
    
    # Generate more extended time variations
    evening_times = ["5:00 PM", "5:30 PM", "6:00 PM", "6:30 PM", "7:00 PM", "7:30 PM", "8:00 PM"]
    weekend_times = ["8:00 AM", "9:00 AM", "10:00 AM", "1:00 PM", "2:00 PM", "3:00 PM"]
    
    for i in range(10):
        subj = random.choice(subjects)
        num = random.choice(course_numbers)
        
        # Evening classes
        evening_time = random.choice(evening_times)
        weekday = random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        extended_samples.append(f"Schedule {subj} {num} {weekday} evening {evening_time}")
        
        # Weekend classes  
        weekend_time = random.choice(weekend_times)
        weekend_day = random.choice(["Saturday", "Sunday"])
        extended_samples.append(f"Add {subj} {num} {weekend_day} {weekend_time}")
    
    # Edge cases and unusual patterns
    edge_case_samples = [
        "Schedule MATH 101 by arrangement",
        "Add CS 301 TBA location MWF 10:00 AM", 
        "Create ENG 201 online delivery Tuesday 2:00 PM",
        "Set PHYS 101 hybrid format TTh 11:00 AM",
        "Book BIOL 301 field work Saturday 8:00 AM",
        "Schedule HIST 401 independent study",
        "Add CHEM 201 make-up session Friday 4:00 PM",
        "Create MATH 301 review session Sunday 7:00 PM",
        "Set ART 101 studio time flexible Friday",
        "Book MUSC 201 practice room Monday 3:00 PM"
    ]

    all_samples = (basic_samples + instructor_samples + advanced_samples + 
                  time_variant_samples + room_variant_samples + complex_samples + 
                  extended_samples + edge_case_samples)

    dataset = []
    for sample in all_samples:
        xml_output = parse_nlp_to_xml(sample)
        dataset.append({
            "context": "TIMETABLE_REQUEST",
            "input": sample, 
            "output": xml_output
        })

    return dataset

def generate_timetable_dataset():
    """Generate and save the timetable dataset"""
    dataset = generate_diverse_dataset()
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

    print(f"✅ Generated {len(dataset)} samples")
    print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Display sample outputs
    for i, sample in enumerate(dataset[:5], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Context: {sample['context']}")
        print(f"Input: {sample['input']}")
        print(f"Output:\n{sample['output']}")
        print("-" * 60)

    print("\n✅ Course timetable dataset generated and saved to data/course_timetable/")

if __name__ == "__main__":
    generate_timetable_dataset()