import json
import random
import os
from typing import List, Dict, Tuple

# Directory to save datasets
output_dir = "data/preference_data"
os.makedirs(output_dir, exist_ok=True)

# Actual values to replace placeholders
DEPARTMENTS = ["CS", "MATH", "PHYS", "CHEM", "ENG", "BIO", "ECON", "HIST", "ENGL", "PSYC"]
BUILDINGS = ["Science Hall", "Engineering Building", "Liberal Arts", "Business Center", "Lab Complex", "Main Hall", "Tech Center", "Research Wing"]
ROOMS = ["101", "102", "205A", "301", "Lab1", "Aud100", "Conf200", "Studio5"]
SUBJECTS = ["CS", "MATH", "PHYS", "CHEM", "BIOL", "ENG", "HIST", "PSYC", "ECON", "PHIL", "STAT", "ART"]
COURSE_IDS = ["101", "102", "150", "201", "202", "250", "301", "302", "350", "401", "402", "450", "499"]
CLASS_TYPES = ["Lecture", "Lab", "Seminar", "Studio", "Workshop", "Tutorial"]
GROUPS = ["Computer Lab", "Science Lab", "Large Lecture", "Small Classroom", "Conference Room"]
FEATURES = ["Projector", "Whiteboard", "Computer", "Audio System", "Video Conference", "Lab Equipment"]
PATTERNS = ["MWF", "TTH", "MW", "WF", "Daily", "Weekend"]
DAYS_OPTIONS = ["MWF", "TTH", "MW", "TR", "F", "S"]
TIMES = ["0800", "0900", "1000", "1100", "1200", "1300", "1400", "1500", "1600", "1700"]
INSTRUCTOR_FIRST = ["John", "Sarah", "Mike", "Lisa", "David", "Anna", "Chris", "Maria", "Tom", "Kate"]
INSTRUCTOR_LAST = ["Smith", "Johnson", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson"]
SUFFIXES = ["A", "B", "C", "1", "2", "3"]
MAX_LOADS = ["12", "15", "18", "20", "24", "30", "36", "40"]

# DTD-compliant distribution types and structures
DISTRIBUTION_TYPES = ['SAME_ROOM', 'SAME_TIME', 'SAME_DAYS', 'DIFFERENT_TIME', 'PRECEDENCE', 'SAME_INSTRUCTOR', 'DIFFERENT_ROOM']
DISTRIBUTION_STRUCTURES = ['AllClasses', 'Progressive', 'GroupsOfTwo', 'GroupsOfThree', 'GroupsOfFour', 'GroupsOfFive', 'Pairwise', 'OneOfEach']

# Preference level mappings for natural language
PREFERENCE_MAPPINGS = {
    "R": ["requires", "must have", "needs", "essential for", "mandatory"],
    "2": ["strongly prefers", "really wants", "loves", "highly desires", "definitely needs"],
    "1": ["prefers", "likes", "wants", "would like", "favors"],
    "0": ["neutral about", "doesn't mind", "okay with", "indifferent to"],
    "-1": ["dislikes", "would rather not", "prefers not", "avoids"],
    "-2": ["strongly dislikes", "hates", "really doesn't want", "strongly avoids"],
    "P": ["prohibited from", "cannot have", "forbidden", "blocked from", "banned from"]
}

def get_valid_time_pair() -> Tuple[str, str]:
    """Get a valid start/end time pair where end > start"""
    start_idx = random.randint(0, len(TIMES)-2)
    end_idx = random.randint(start_idx+1, len(TIMES)-1)
    return TIMES[start_idx], TIMES[end_idx]

def get_random_level_and_text() -> Tuple[str, str]:
    """Get a random preference level and corresponding natural language"""
    level = random.choice(["R", "-2", "-1", "0", "1", "2", "P"])
    text = random.choice(PREFERENCE_MAPPINGS[level])
    return level, text

def generate_consistent_values() -> Dict[str, str]:
    """Generate consistent values that will be reused across input/output"""
    start_time, end_time = get_valid_time_pair()
    
    return {
        "subject": random.choice(SUBJECTS),
        "course_id": random.choice(COURSE_IDS),
        "department": random.choice(DEPARTMENTS),
        "building": random.choice(BUILDINGS),
        "room": random.choice(ROOMS),
        "feature": random.choice(FEATURES),
        "class_type": random.choice(CLASS_TYPES),
        "group": random.choice(GROUPS),
        "pattern": random.choice(PATTERNS),
        "days": random.choice(DAYS_OPTIONS),
        "start_time": start_time,
        "end_time": end_time,
        "instructor_first": random.choice(INSTRUCTOR_FIRST),
        "instructor_last": random.choice(INSTRUCTOR_LAST),
        "suffix": random.choice(SUFFIXES),
        "max_load": random.choice(MAX_LOADS),
        "dist_type": random.choice(DISTRIBUTION_TYPES),
        "dist_structure": random.choice(DISTRIBUTION_STRUCTURES)
    }

def generate_natural_variations():
    """Generate natural preference request language"""
    starters = [
        "Set preferences for",
        "Configure preferences for",
        "I need to set up preferences for",
        "Please configure preferences for",
        "Set scheduling preferences for",
        "Help me configure",
        "I want to set preferences for",
        "Can you configure",
        "Please set up preferences for",
        "Configure the schedule preferences for"
    ]
    return random.choice(starters)

def generate_single_intent_example() -> Dict:
    """Generate single preference examples following DTD structure"""
    values = generate_consistent_values()
    element_type = random.choice(["department", "instructor", "subpart", "class"])
    
    # Choose appropriate preference types based on DTD
    if element_type == "department":
        pref_type = random.choice(["timePref", "roomPref", "buildingPref", "groupPref", "featurePref"])
    elif element_type == "instructor":
        pref_type = random.choice(["timePref", "coursePref", "teachingPref", "roomPref", "buildingPref"])
    else:  # subpart or class
        pref_type = random.choice(["timePref", "roomPref", "buildingPref", "groupPref", "featurePref"])
    
    level, level_text = get_random_level_and_text()
    starter = generate_natural_variations()
    
    # Generate examples based on element and preference type
    if element_type == "department" and pref_type == "timePref":
        input_text = f"{starter} {values['department']} department - they {level_text} classes on {values['days']} from {values['start_time']} to {values['end_time']}."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="{values['department']}">
    <timePref level="{level}">
      <pref days="{values['days']}" start="{values['start_time']}" stop="{values['end_time']}" level="{level}"/>
    </timePref>
  </department>
</preferences>"""
    
    elif element_type == "department" and pref_type == "buildingPref":
        input_text = f"The {values['department']} department {level_text} using {values['building']} for their classes."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="{values['department']}">
    <buildingPref building="{values['building']}" level="{level}"/>
  </department>
</preferences>"""
    
    elif element_type == "department" and pref_type == "featurePref":
        input_text = f"{starter} {values['department']} department - they {level_text} {values['feature']} in their classrooms."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="{values['department']}">
    <featurePref feature="{values['feature']}" level="{level}"/>
  </department>
</preferences>"""
    
    elif element_type == "instructor" and pref_type == "coursePref":
        input_text = f"Professor {values['instructor_first']} {values['instructor_last']} {level_text} teaching {values['subject']} {values['course_id']} this semester."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="{values['instructor_first']}" lastName="{values['instructor_last']}" department="{values['department']}">
    <coursePref subject="{values['subject']}" course="{values['course_id']}" level="{level}"/>
  </instructor>
</preferences>"""
    
    elif element_type == "instructor" and pref_type == "teachingPref":
        input_text = f"{starter} {values['instructor_first']} {values['instructor_last']} - they {level_text} teaching {values['max_load']} credit hours maximum."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="{values['instructor_first']}" lastName="{values['instructor_last']}" department="{values['department']}">
    <teachingPref maxLoad="{values['max_load']}" level="{level}"/>
  </instructor>
</preferences>"""
    
    elif element_type == "instructor" and pref_type == "timePref":
        input_text = f"{starter} {values['instructor_first']} {values['instructor_last']}'s schedule - they {level_text} {values['days']} time slots."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="{values['instructor_first']}" lastName="{values['instructor_last']}" department="{values['department']}">
    <timePref level="{level}">
      <pref days="{values['days']}" start="{values['start_time']}" stop="{values['end_time']}" level="{level}"/>
    </timePref>
  </instructor>
</preferences>"""
    
    elif element_type == "class" and pref_type == "roomPref":
        input_text = f"{values['subject']} {values['course_id']} section {values['suffix']} {level_text} being scheduled in room {values['room']} at {values['building']}."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <class subject="{values['subject']}" course="{values['course_id']}" suffix="{values['suffix']}" type="{values['class_type']}">
    <roomPref building="{values['building']}" room="{values['room']}" level="{level}"/>
  </class>
</preferences>"""
    
    elif element_type == "subpart" and pref_type == "featurePref":
        input_text = f"{starter} {values['subject']} {values['course_id']} {values['class_type']} - they {level_text} {values['feature']} equipment."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="{values['subject']}" course="{values['course_id']}" type="{values['class_type']}">
    <featurePref feature="{values['feature']}" level="{level}"/>
  </subpart>
</preferences>"""
    
    elif element_type == "subpart" and pref_type == "groupPref":
        input_text = f"{starter} {values['subject']} {values['course_id']} course - they {level_text} {values['group']} type rooms."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="{values['subject']}" course="{values['course_id']}" type="{values['class_type']}">
    <groupPref group="{values['group']}" level="{level}"/>
  </subpart>
</preferences>"""
    
    else:  # Default subpart timePref
        input_text = f"{starter} {values['subject']} {values['course_id']} {values['class_type']} - they {level_text} {values['days']} time slots."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="{values['subject']}" course="{values['course_id']}" type="{values['class_type']}">
    <timePref level="{level}">
      <pref days="{values['days']}" start="{values['start_time']}" stop="{values['end_time']}" level="{level}"/>
    </timePref>
  </subpart>
</preferences>"""
    
    return {
        "context": "PREFERENCE_REQUEST",
        "input": input_text,
        "output": xml
    }

def generate_merged_example() -> Dict:
    """Generate complex examples with multiple preferences using consistent values"""
    values = generate_consistent_values()
    level1, level_text1 = get_random_level_and_text()
    level2, level_text2 = get_random_level_and_text()
    
    scenarios = [
        {
            "input": f"Set up {values['department']} department preferences: {level_text1} {values['building']} and instructor {values['instructor_first']} {values['instructor_last']} {level_text2} morning classes.",
            "output": f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="{values['department']}">
    <buildingPref building="{values['building']}" level="{level1}"/>
  </department>
  <instructor firstName="{values['instructor_first']}" lastName="{values['instructor_last']}" department="{values['department']}">
    <timePref level="{level2}">
      <pref days="MWF" start="0800" stop="1200" level="{level2}"/>
    </timePref>
  </instructor>
</preferences>"""
        },
        {
            "input": f"Configure {values['subject']} {values['course_id']} course: {level_text1} {values['feature']} equipment and {level_text2} room {values['room']}.",
            "output": f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="{values['subject']}" course="{values['course_id']}" type="{values['class_type']}">
    <featurePref feature="{values['feature']}" level="{level1}"/>
    <roomPref building="{values['building']}" room="{values['room']}" level="{level2}"/>
  </subpart>
</preferences>"""
        },
        {
            "input": f"Set preferences for {values['subject']} {values['course_id']} section {values['suffix']}: {level_text1} {values['days']} schedule and {level_text2} {values['group']} rooms.",
            "output": f"""<preferences term="Fall" year="2010" campus="woebegon">
  <class subject="{values['subject']}" course="{values['course_id']}" suffix="{values['suffix']}" type="{values['class_type']}">
    <timePref level="{level1}">
      <pref days="{values['days']}" start="{values['start_time']}" stop="{values['end_time']}" level="{level1}"/>
    </timePref>
    <groupPref group="{values['group']}" level="{level2}"/>
  </class>
</preferences>"""
        },
        {
            "input": f"Configure instructor {values['instructor_first']} {values['instructor_last']}: {level_text1} teaching {values['subject']} {values['course_id']} and {level_text2} {values['max_load']} credit hours max.",
            "output": f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="{values['instructor_first']}" lastName="{values['instructor_last']}" department="{values['department']}">
    <coursePref subject="{values['subject']}" course="{values['course_id']}" level="{level1}"/>
    <teachingPref maxLoad="{values['max_load']}" level="{level2}"/>
  </instructor>
</preferences>"""
        }
    ]
    
    scenario = random.choice(scenarios)
    return {
        "context": "PREFERENCE_REQUEST",
        "input": scenario["input"],
        "output": scenario["output"]
    }

def generate_dataset(n_single: int, n_merged: int) -> List[Dict]:
    """Generate complete dataset with both single and merged examples"""
    dataset = []
    
    for _ in range(n_single):
        dataset.append(generate_single_intent_example())
    
    for _ in range(n_merged):
        dataset.append(generate_merged_example())
    
    random.shuffle(dataset)
    return dataset

def split_dataset(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Split dataset into train/val/test"""
    random.shuffle(data)
    n = len(data)
    return {
        "train_pref": data[:int(0.7 * n)],
        "val_pref": data[int(0.7 * n):int(0.85 * n)],
        "test_pref": data[int(0.85 * n):]
    }

def save_dataset(dataset: Dict[str, List[Dict]]):
    """Save dataset splits to JSON files"""
    for split, data in dataset.items():
        path = os.path.join(output_dir, f"{split}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

def generate_preference_dataset():
    """Main function to generate the complete preference dataset"""
    dataset = generate_dataset(n_single=800, n_merged=400)
    split_data = split_dataset(dataset)
    save_dataset(split_data)
    
    print(f"Generated {len(dataset)} examples with consistent values")
    print(f"Train: {len(split_data['train_pref'])}, Val: {len(split_data['val_pref'])}, Test: {len(split_data['test_pref'])}")
    
    # Show sample examples
    print("\nSample examples:")
    for i, example in enumerate(dataset[:3]):
        print(f"\nExample {i+1}:")
        print(f"Context: {example['context']}")
        print(f"Input: {example['input']}")
        print(f"Output: {example['output'][:200]}...")

# if __name__ == "__main__":
#     generate_preference_dataset()