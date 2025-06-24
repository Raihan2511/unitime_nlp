import json
import random
import os
from typing import List, Dict

# Directory to save datasets
output_dir = "data/preference_data"
os.makedirs(output_dir, exist_ok=True)

# Actual values to replace placeholders
DEPARTMENTS = ["CS", "MATH", "PHYS", "CHEM", "ENG", "BIO", "ECON", "HIST", "ENGL", "PSYC"]
BUILDINGS = ["Science Hall", "Engineering Building", "Liberal Arts", "Business Center", "Lab Complex", "Main Hall", "Tech Center", "Research Wing"]
ROOMS = ["101", "102", "205A", "301", "Lab1", "Aud100", "Conf200", "Studio5"]
SUBJECTS = ["CS", "MATH", "PHYS", "CHEM", "BIOL", "ENG", "HIST", "PSYC", "ECON", "PHIL", "STAT", "ART"]
COURSE_IDS =["101", "102", "150", "201", "202", "250", "301", "302", "350", "401", "402", "450", "499"]
CLASS_TYPES = ["Lecture", "Lab", "Seminar", "Studio", "Workshop", "Tutorial"]
GROUPS = ["Computer Lab", "Science Lab", "Large Lecture", "Small Classroom", "Conference Room"]
FEATURES = ["Projector", "Whiteboard", "Computer", "Audio System", "Video Conference", "Lab Equipment"]
PATTERNS = ["MWF", "TTH", "MW", "WF", "Daily", "Weekend"]
DAYS_OPTIONS = ["MWF", "TTH", "MW", "TR", "F", "S"]
TIMES = ["0800", "0900", "1000", "1100", "1300", "1400", "1500", "1600", "1700"]
INSTRUCTOR_FIRST = ["John", "Sarah", "Mike", "Lisa", "David", "Anna", "Chris", "Maria", "Tom", "Kate"]
INSTRUCTOR_LAST = ["Smith", "Johnson", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson"]
SUFFIXES = ["A", "B", "C", "1", "2", "3"]
MAX_LOADS = ["12", "15", "18", "20", "24","50","30", "36", "40", "45", "60"]

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

DISTRIBUTION_TYPES = ['SAME_ROOM', 'SAME_TIME', 'SAME_DAYS', 'DIFFERENT_TIME', 'PRECEDENCE', 'SAME_INSTRUCTOR', 'DIFFERENT_ROOM']
DISTRIBUTION_STRUCTURES = ['AllClasses', 'Progressive', 'GroupsOfTwo', 'GroupsOfThree', 'Pairwise']

def replace_placeholders(text: str) -> str:
    """Replace all placeholders with actual values"""
    replacements = {
        "[DEPARTMENT]": random.choice(DEPARTMENTS),
        "[BUILDING]": random.choice(BUILDINGS),
        "[ROOM_ID]": random.choice(ROOMS),
        "[SUBJECT]": random.choice(SUBJECTS),
        "[COURSE_ID]": random.choice(COURSE_IDS),
        "[CLASS_TYPE]": random.choice(CLASS_TYPES),
        "[GROUP]": random.choice(GROUPS),
        "[FEATURE]": random.choice(FEATURES),
        "[PATTERN]": random.choice(PATTERNS),
        "[DAYS]": random.choice(DAYS_OPTIONS),
        "[START_TIME]": random.choice(TIMES),
        "[END_TIME]": random.choice(TIMES[1:]),  # Avoid same start/end
        "[INSTRUCTOR_NAME]": random.choice(INSTRUCTOR_FIRST),
        "[LAST_NAME]": random.choice(INSTRUCTOR_LAST),
        "[SUFFIX]": random.choice(SUFFIXES),
        "[MAX_LOAD]": random.choice(MAX_LOADS)
    }
    
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text

def get_random_level_and_text():
    level = random.choice(["R", "-2", "-1", "0", "1", "2", "P"])
    text = random.choice(PREFERENCE_MAPPINGS[level])
    return level, text


def generate_natural_variations():
    """Generate more natural conversation starters"""
    starters = [
        "Can you help me set up",
        "I need to configure",
        "Please schedule",
        "We want to arrange",
        "Could you organize",
        "I would like to set preferences for",
        "Help me with scheduling",
        "Set up the timetable for",
        "Schedule a session for",
        "Arrange classes for",
        "I want to create a schedule",
        "Can you help with my timetable",
        "Assign courses for",
        "Plan the week for",
        # "Let’s organize the semester",
        # "I’d like to plan the academic term",
        "Get the classes arranged for",
        "Prepare a plan for",
        "Configure the academic setup for"
    ]
    return random.choice(starters)

def generate_single_intent_example() -> Dict:
    element_type = random.choice(["department", "instructor", "subpart", "class"])
    
    if element_type == "department":
        pref_type = random.choice(["timePref", "roomPref", "buildingPref", "groupPref", "featurePref"])
    elif element_type == "instructor":
        pref_type = random.choice(["timePref", "coursePref", "teachingPref", "roomPref", "buildingPref"])
    else:
        pref_type = random.choice(["timePref", "roomPref", "buildingPref", "groupPref", "featurePref"])
    
    level, level_text = get_random_level_and_text()
    starter = generate_natural_variations()
    
    if element_type == "department" and pref_type == "timePref":
        input_text = f"{starter} [DEPARTMENT] department - they {level_text} classes on [DAYS] from [START_TIME] to [END_TIME]."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    <timePref level="{level}">
      <pref days="[DAYS]" start="[START_TIME]" stop="[END_TIME]" level="{level}"/>
    </timePref>
  </department>
</preferences>"""
    
    elif element_type == "department" and pref_type == "buildingPref":
        input_text = f"The [DEPARTMENT] department {level_text} using [BUILDING] for their classes."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    <buildingPref building="[BUILDING]" level="{level}"/>
  </department>
</preferences>"""
    
    elif element_type == "instructor" and pref_type == "coursePref":
        input_text = f"Professor [INSTRUCTOR_NAME] [LAST_NAME] {level_text} teaching [SUBJECT] [COURSE_ID] this semester."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <coursePref subject="[SUBJECT]" course="[COURSE_ID]" level="{level}"/>
  </instructor>
</preferences>"""
    
    elif element_type == "instructor" and pref_type == "timePref":
        input_text = f"{starter} [INSTRUCTOR_NAME] [LAST_NAME]'s schedule - {level_text} [DAYS] time slots."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <timePref level="{level}">
      <pref days="[DAYS]" start="[START_TIME]" stop="[END_TIME]" level="{level}"/>
    </timePref>
  </instructor>
</preferences>"""
    
    elif element_type == "class" and pref_type == "roomPref":
        input_text = f"[SUBJECT] [COURSE_ID] section [SUFFIX] {level_text} being scheduled in room [ROOM_ID] at [BUILDING]."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <class subject="[SUBJECT]" course="[COURSE_ID]" suffix="[SUFFIX]" type="[CLASS_TYPE]">
    <roomPref building="[BUILDING]" room="[ROOM_ID]" level="{level}"/>
  </class>
</preferences>"""
    
    else:  # Default case
        input_text = f"{starter} [SUBJECT] [COURSE_ID] [CLASS_TYPE] with {level_text} [FEATURE] equipment."
        xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]">
    <featurePref feature="[FEATURE]" level="{level}"/>
  </subpart>
</preferences>"""
    
    # Replace placeholders with actual values
    input_text = replace_placeholders(input_text)
    xml = replace_placeholders(xml)
    
    return {"input": input_text, "output": xml}

def generate_merged_example() -> Dict:
    """Generate complex examples with multiple preferences"""
    level1, level_text1 = get_random_level_and_text()
    level2, level_text2 = get_random_level_and_text()
    
    scenarios = [
        {
            "input": f"Set up [DEPARTMENT] department preferences: {level_text1} [BUILDING] and instructor [INSTRUCTOR_NAME] [LAST_NAME] {level_text2} morning classes.",
            "output": f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    <buildingPref building="[BUILDING]" level="{level1}"/>
  </department>
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <timePref level="{level2}">
      <pref days="MWF" start="0800" stop="1200" level="{level2}"/>
    </timePref>
  </instructor>
</preferences>"""
        },
        {
            "input": f"Configure [SUBJECT] [COURSE_ID] course: {level_text1} [FEATURE] equipment and {level_text2} room [ROOM_ID].",
            "output": f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]">
    <featurePref feature="[FEATURE]" level="{level1}"/>
    <roomPref building="[BUILDING]" room="[ROOM_ID]" level="{level2}"/>
  </subpart>
</preferences>"""
        }
    ]
    
    scenario = random.choice(scenarios)
    return {
        "input": replace_placeholders(scenario["input"]),
        "output": replace_placeholders(scenario["output"])
    }

def generate_dataset(n_single: int, n_merged: int) -> List[Dict]:
    dataset = []
    
    for _ in range(n_single):
        dataset.append(generate_single_intent_example())
    
    for _ in range(n_merged):
        dataset.append(generate_merged_example())
    
    random.shuffle(dataset)
    return dataset

def split_dataset(data: List[Dict]) -> Dict[str, List[Dict]]:
    random.shuffle(data)
    n = len(data)
    return {
        "train_pref": data[:int(0.7 * n)],
        "val_pref": data[int(0.7 * n):int(0.85 * n)],
        "test_pref": data[int(0.85 * n):]
    }

def save_dataset(dataset: Dict[str, List[Dict]]):
    for split, data in dataset.items():
        path = os.path.join(output_dir, f"{split}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

def generate_preference_dataset():
    dataset = generate_dataset(n_single=600, n_merged=400)
    split_data = split_dataset(dataset)
    save_dataset(split_data)
    
    print(f"Generated {len(dataset)} examples with actual values")
    print(f"Train: {len(split_data['train_pref'])}, Val: {len(split_data['val_pref'])}, Test: {len(split_data['test_pref'])}")
    
    # Show sample
    print("\nSample examples:")
    for i, example in enumerate(dataset[:2]):
        print(f"\nExample {i+1}:")
        print(f"Input: {example['input']}")
        print(f"Output: {example['output'][:200]}...")

# if __name__ == "__main__":
#     generate_preference_dataset()