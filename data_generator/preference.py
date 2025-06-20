import json
import random
import os
from typing import List, Dict, Tuple

# Directory to save datasets
output_dir = "data/preference_data"
os.makedirs(output_dir, exist_ok=True)

# Preference level mappings based on DTD
PREFERENCE_LEVELS = {
    'R': 'required',
    'P': 'preferred', 
    '2': 'strongly preferred',
    '1': 'preferred',
    '0': 'neutral',
    '-1': 'discouraged',
    '-2': 'strongly discouraged'
}

# Sample data for realistic generation
BUILDINGS = ['EDUC', 'THTR', 'COMP', 'BIOL', 'CHEM', 'MATH', 'PHYS']
ROOMS = ['101', '102', '103', '104', '105', '106', '107', '108', '201', '202']
SUBJECTS = ['MATH', 'CHEM', 'BIOL', 'PHYS', 'ENGL', 'HIST', 'COMP', 'ECON', 'PSY']
COURSE_NUMBERS = ['101', '102', '201', '202', '301', '302']
CLASS_TYPES = ['Lec', 'Lab', 'Rec', 'Sem']
DAYS = ['M', 'T', 'W', 'R', 'F']
DAY_COMBINATIONS = ['MWF', 'TR', 'MW', 'WF', 'MTWRF']
TIMES = ['0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600']
PATTERNS = ['3 x 50', '2 x 75', '1 x 100', '1 x 150', '5 x 50']
GROUPS = ['Classroom', 'Lab', 'Comp Labs', 'Biol Labs', 'Chem Labs']
FEATURES = ['Projector', 'Whiteboard', 'Computer', 'Lab Equipment', 'Theater Seating']
INSTRUCTOR_NAMES = [
    ('JOHN', 'SMITH'), ('MARY', 'JOHNSON'), ('DAVID', 'WILLIAMS'), 
    ('SARAH', 'BROWN'), ('MICHAEL', 'JONES'), ('LISA', 'GARCIA')
]

def get_preference_level_text(level: str) -> str:
    """Convert preference level to natural language"""
    return PREFERENCE_LEVELS.get(level, 'neutral')

def generate_department_preference() -> Dict:
    """Generate department-level preferences"""
    dept_code = f"010{random.randint(0, 9)}"
    
    # Choose preference type
    pref_types = ['room', 'building', 'distribution']
    pref_type = random.choice(pref_types)
    
    if pref_type == 'room':
        building = random.choice(BUILDINGS)
        room = random.choice(ROOMS)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"Department {dept_code} {level_text} room {room} in {building} building."
        xml = f'<department code="{dept_code}">\n  <roomPref building="{building}" room="{room}" level="{level}"/>\n</department>'
    
    elif pref_type == 'building':
        building = random.choice(BUILDINGS)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"Department {dept_code} {level_text} {building} building."
        xml = f'<department code="{dept_code}">\n  <buildingPref building="{building}" level="{level}"/>\n</department>'
    
    else:  # distribution
        subject = random.choice(SUBJECTS)
        course = random.choice(COURSE_NUMBERS)
        class_type = random.choice(CLASS_TYPES)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"Department {dept_code} requires {subject} {course} {class_type} classes to be scheduled in the same room."
        xml = f'''<department code="{dept_code}">
  <distributionPref type="SAME_ROOM" structure="Progressive" level="{level}">
    <subpart subject="{subject}" course="{course}" type="{class_type}"/>
  </distributionPref>
</department>'''
    
    return {"input": input_text, "output": xml}

def generate_instructor_preference() -> Dict:
    """Generate instructor preferences"""
    first_name, last_name = random.choice(INSTRUCTOR_NAMES)
    dept_code = f"010{random.randint(0, 9)}"
    external_id = str(random.randint(100, 999))
    
    # Choose preference type
    pref_types = ['time', 'building', 'feature', 'course', 'teaching_load']
    pref_type = random.choice(pref_types)
    
    if pref_type == 'time':
        day = random.choice(DAYS)
        start_time = random.choice(TIMES)
        end_time = str(int(start_time) + 200)  # 2 hours later
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        day_names = {'M': 'Monday', 'T': 'Tuesday', 'W': 'Wednesday', 'R': 'Thursday', 'F': 'Friday'}
        
        input_text = f"Instructor {first_name} {last_name} {level_text} teaching on {day_names[day]} from {start_time[:2]}:{start_time[2:]} to {end_time[:2]}:{end_time[2:]}."
        xml = f'''<instructor externalId="{external_id}" firstName="{first_name}" lastName="{last_name}" department="{dept_code}">
  <timePref level="R">
    <pref level="{level}" day="{day}" start="{start_time}" stop="{end_time}"/>
  </timePref>
</instructor>'''
    
    elif pref_type == 'building':
        building = random.choice(BUILDINGS)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"Instructor {first_name} {last_name} {level_text} {building} building."
        xml = f'''<instructor externalId="{external_id}" firstName="{first_name}" lastName="{last_name}" department="{dept_code}">
  <buildingPref building="{building}" level="{level}"/>
</instructor>'''
    
    elif pref_type == 'feature':
        feature = random.choice(FEATURES)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"Instructor {first_name} {last_name} {level_text} rooms with {feature}."
        xml = f'''<instructor externalId="{external_id}" firstName="{first_name}" lastName="{last_name}" department="{dept_code}">
  <featurePref feature="{feature}" level="{level}"/>
</instructor>'''
    
    elif pref_type == 'course':
        subject = random.choice(SUBJECTS)
        course = random.choice(COURSE_NUMBERS)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"Instructor {first_name} {last_name} {level_text} teaching {subject} {course}."
        xml = f'''<instructor externalId="{external_id}" firstName="{first_name}" lastName="{last_name}" department="{dept_code}">
  <coursePref subject="{subject}" course="{course}" level="{level}"/>
</instructor>'''
    
    else:  # teaching_load
        max_load = random.choice(['12.0', '15.0', '18.0', '20.0'])
        level = random.choice(['0', '1', '2'])
        level_text = get_preference_level_text(level)
        
        input_text = f"Instructor {first_name} {last_name} {level_text} maximum teaching load of {max_load} hours."
        xml = f'''<instructor externalId="{external_id}" firstName="{first_name}" lastName="{last_name}" department="{dept_code}">
  <teachingPref maxLoad="{max_load}" level="{level}"/>
</instructor>'''
    
    return {"input": input_text, "output": xml}

def generate_subpart_preference() -> Dict:
    """Generate subpart preferences"""
    subject = random.choice(SUBJECTS)
    course = random.choice(COURSE_NUMBERS)
    class_type = random.choice(CLASS_TYPES)
    
    # Choose preference type
    pref_types = ['time_pattern', 'group', 'room']
    pref_type = random.choice(pref_types)
    
    if pref_type == 'time_pattern':
        pattern = random.choice(PATTERNS)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"{subject} {course} {class_type} {level_text} {pattern} time pattern."
        xml = f'''<subpart subject="{subject}" course="{course}" type="{class_type}">
  <timePref pattern="{pattern}" level="{level}"/>
</subpart>'''
    
    elif pref_type == 'group':
        group = random.choice(GROUPS)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"{subject} {course} {class_type} {level_text} {group} group."
        xml = f'''<subpart subject="{subject}" course="{course}" type="{class_type}">
  <groupPref group="{group}" level="{level}"/>
</subpart>'''
    
    else:  # room
        building = random.choice(BUILDINGS)
        room = random.choice(ROOMS)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"{subject} {course} {class_type} {level_text} room {room} in {building}."
        xml = f'''<subpart subject="{subject}" course="{course}" type="{class_type}">
  <roomPref building="{building}" room="{room}" level="{level}"/>
</subpart>'''
    
    return {"input": input_text, "output": xml}

def generate_class_preference() -> Dict:
    """Generate class preferences"""
    subject = random.choice(SUBJECTS)
    course = random.choice(COURSE_NUMBERS)
    class_type = random.choice(CLASS_TYPES)
    suffix = str(random.randint(1, 12))
    
    # Choose preference type
    pref_types = ['time_specific', 'room', 'building']
    pref_type = random.choice(pref_types)
    
    if pref_type == 'time_specific':
        days = random.choice(DAY_COMBINATIONS)
        time = random.choice(TIMES)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"Class {subject} {course} {class_type} section {suffix} {level_text} {days} at {time[:2]}:{time[2:]}."
        xml = f'''<class subject="{subject}" course="{course}" type="{class_type}" suffix="{suffix}">
  <timePref level="R">
    <pref level="{level}" days="{days}" time="{time}"/>
  </timePref>
</class>'''
    
    elif pref_type == 'room':
        building = random.choice(BUILDINGS)
        room = random.choice(ROOMS)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"Class {subject} {course} {class_type} section {suffix} {level_text} room {room} in {building}."
        xml = f'''<class subject="{subject}" course="{course}" type="{class_type}" suffix="{suffix}">
  <roomPref building="{building}" room="{room}" level="{level}"/>
</class>'''
    
    else:  # building
        building = random.choice(BUILDINGS)
        level = random.choice(list(PREFERENCE_LEVELS.keys()))
        level_text = get_preference_level_text(level)
        
        input_text = f"Class {subject} {course} {class_type} section {suffix} {level_text} {building} building."
        xml = f'''<class subject="{subject}" course="{course}" type="{class_type}" suffix="{suffix}">
  <buildingPref building="{building}" level="{level}"/>
</class>'''
    
    return {"input": input_text, "output": xml}

def generate_complex_preference() -> Dict:
    """Generate complex multi-element preferences"""
    scenarios = ['instructor_course_time', 'department_distribution', 'linked_classes']
    scenario = random.choice(scenarios)
    
    if scenario == 'instructor_course_time':
        first_name, last_name = random.choice(INSTRUCTOR_NAMES)
        subject = random.choice(SUBJECTS)
        course = random.choice(COURSE_NUMBERS)
        class_type = random.choice(CLASS_TYPES)
        building = random.choice(BUILDINGS)
        room = random.choice(ROOMS)
        days = random.choice(DAY_COMBINATIONS)
        time = random.choice(TIMES)
        
        input_text = f"Instructor {first_name} {last_name} prefers to teach {subject} {course} {class_type} in room {room} of {building} building on {days} at {time[:2]}:{time[2:]}."
        xml = f'''<preferences term="Fal" year="2025" campus="woebegon">
  <instructor firstName="{first_name}" lastName="{last_name}" department="010{random.randint(0,9)}">
    <coursePref subject="{subject}" course="{course}" level="P"/>
    <buildingPref building="{building}" level="1"/>
  </instructor>
  <subpart subject="{subject}" course="{course}" type="{class_type}">
    <roomPref building="{building}" room="{room}" level="P"/>
    <timePref level="R">
      <pref level="P" days="{days}" time="{time}"/>
    </timePref>
  </subpart>
</preferences>'''
    
    elif scenario == 'department_distribution':
        dept_code = f"010{random.randint(0, 9)}"
        subject = random.choice(SUBJECTS)
        course = random.choice(COURSE_NUMBERS)
        
        input_text = f"Department {dept_code} requires {subject} {course} Lecture, Lab, and Recitation to be scheduled in the same room."
        xml = f'''<preferences term="Fal" year="2025" campus="woebegon">
  <department code="{dept_code}">
    <distributionPref type="SAME_ROOM" structure="Progressive" level="-1">
      <subpart subject="{subject}" course="{course}" type="Lec"/>
      <subpart subject="{subject}" course="{course}" type="Lab"/>
      <subpart subject="{subject}" course="{course}" type="Rec"/>
    </distributionPref>
  </department>
</preferences>'''
    
    else:  # linked_classes
        subject = random.choice(SUBJECTS)
        course = random.choice(COURSE_NUMBERS)
        suffixes = [str(i) for i in range(1, 4)]
        
        input_text = f"All {subject} {course} Recitation sections must be scheduled on the same days."
        xml = f'''<preferences term="Fal" year="2025" campus="woebegon">
  <department code="010{random.randint(0,9)}">
    <distributionPref type="SAME_DAYS" structure="AllClasses" level="-2">
      <class subject="{subject}" course="{course}" type="Rec" suffix="{suffixes[0]}"/>
      <class subject="{subject}" course="{course}" type="Rec" suffix="{suffixes[1]}"/>
      <class subject="{subject}" course="{course}" type="Rec" suffix="{suffixes[2]}"/>
    </distributionPref>
  </department>
</preferences>'''
    
    return {"input": input_text, "output": xml}

def generate_single_intent_example() -> Dict:
    """Generate single-intent examples"""
    generators = [
        generate_department_preference,
        generate_instructor_preference, 
        generate_subpart_preference,
        generate_class_preference
    ]
    return random.choice(generators)()

def generate_merged_example() -> Dict:
    """Generate complex multi-intent examples"""
    return generate_complex_preference()

def generate_dataset(n_single: int, n_merged: int) -> List[Dict]:
    """Generate complete dataset"""
    dataset = []
    
    # Generate single-intent examples
    for _ in range(n_single):
        dataset.append(generate_single_intent_example())
    
    # Generate merged/complex examples
    for _ in range(n_merged):
        dataset.append(generate_merged_example())
    
    random.shuffle(dataset)
    return dataset

def split_dataset(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Split dataset into train/val/test"""
    random.shuffle(data)
    n = len(data)
    return {
        "train": data[:int(0.7 * n)],
        "val": data[int(0.7 * n):int(0.85 * n)],
        "test": data[int(0.85 * n):]
    }

def save_dataset(dataset: Dict[str, List[Dict]]):
    """Save dataset to JSON files"""
    for split, data in dataset.items():
        path = os.path.join(output_dir, f"{split}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} examples to {path}")

def generate_preference_dataset():
    """Main function to generate the complete dataset"""
    print("Generating preference dataset...")
    dataset = generate_dataset(n_single=200, n_merged=100)
    split_data = split_dataset(dataset)
    save_dataset(split_data)
    print(f"Dataset generation complete! Total examples: {len(dataset)}")
    
    # Print sample examples
    print("\nSample examples:")
    for i, example in enumerate(dataset[:3]):
        print(f"\nExample {i+1}:")
        print(f"Input: {example['input']}")
        print(f"Output: {example['output']}")

# if __name__ == "__main__":
#     generate_preference_dataset()