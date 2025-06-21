import json
import random
import os
from typing import List, Dict

# Directory to save datasets
output_dir = "data/preference_data"
os.makedirs(output_dir, exist_ok=True)

# Preference level mappings for natural language
PREFERENCE_MAPPINGS = {
    "R": ["requires", "must have", "needs", "essential", "mandatory"],
    "2": ["strongly prefers", "really wants", "highly desires", "loves"],
    "1": ["prefers", "likes", "wants", "would like", "favors"],
    "0": ["neutral about", "doesn't mind", "indifferent to", "okay with"],
    "-1": ["dislikes", "would rather not", "prefers not", "avoids"],
    "-2": ["strongly dislikes", "really doesn't want", "hates", "strongly avoids"],
    "P": ["prohibited from", "cannot have", "forbidden", "blocked from", "banned from"]
}

# Distribution types and structures from DTD
DISTRIBUTION_TYPES = ['SAME_ROOM', 'SAME_TIME', 'SAME_DAYS', 'DIFFERENT_TIME', 'PRECEDENCE', 'SAME_INSTRUCTOR', 'DIFFERENT_ROOM']
DISTRIBUTION_STRUCTURES = ['AllClasses', 'Progressive', 'GroupsOfTwo', 'GroupsOfThree', 'GroupsOfFour', 'GroupsOfFive', 'Pairwise', 'OneOfEach']

# All preference types that can be used in each element
DEPARTMENT_PREFS = ["timePref", "roomPref", "buildingPref", "groupPref", "featurePref", "datePref", "distributionPref"]
INSTRUCTOR_PREFS = ["timePref", "roomPref", "buildingPref", "groupPref", "featurePref", "datePref", "distributionPref", "coursePref", "teachingPref"]
SUBPART_PREFS = ["timePref", "roomPref", "buildingPref", "groupPref", "featurePref", "datePref"]
CLASS_PREFS = ["timePref", "roomPref", "buildingPref", "groupPref", "featurePref", "datePref"]

def get_random_level_and_text():
    level = random.choice(["R", "-2", "-1", "0", "1", "2", "P"])
    text = random.choice(PREFERENCE_MAPPINGS[level])
    return level, text

def generate_timePref_xml(level):
    """Generate timePref with nested pref elements"""
    pref_level, _ = get_random_level_and_text()
    return f"""<timePref level="{level}">
    <pref days="[DAYS]" start="[START_TIME]" stop="[END_TIME]" level="{pref_level}"/>
  </timePref>"""

def generate_distributionPref_xml(level, element_type="class"):
    """Generate distributionPref with nested class/subpart elements"""
    dist_type = random.choice(DISTRIBUTION_TYPES)
    structure = random.choice(DISTRIBUTION_STRUCTURES)
    
    if element_type == "class":
        nested = '<class subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]"/>'
    else:
        nested = '<subpart subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]"/>'
    
    return f"""<distributionPref type="{dist_type}" structure="{structure}" level="{level}">
    {nested}
  </distributionPref>"""

def generate_single_intent_example() -> Dict:
    # Choose element type and preference type
    element_type = random.choice(["department", "instructor", "subpart", "class"])
    
    if element_type == "department":
        pref_type = random.choice(DEPARTMENT_PREFS)
    elif element_type == "instructor":
        pref_type = random.choice(INSTRUCTOR_PREFS)
    elif element_type == "subpart":
        pref_type = random.choice(SUBPART_PREFS)
    else:  # class
        pref_type = random.choice(CLASS_PREFS)
    
    level, level_text = get_random_level_and_text()
    
    # Generate based on element + preference combination
    if element_type == "department":
        if pref_type == "timePref":
            input_text = f"Department [DEPARTMENT] {level_text} scheduling classes on [DAYS] from [START_TIME] to [END_TIME]."
            pref_xml = generate_timePref_xml(level)
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    {pref_xml}
  </department>
</preferences>"""
        
        elif pref_type == "roomPref":
            input_text = f"Department [DEPARTMENT] {level_text} room [ROOM_ID] in building [BUILDING]."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    <roomPref building="[BUILDING]" room="[ROOM_ID]" level="{level}"/>
  </department>
</preferences>"""
        
        elif pref_type == "buildingPref":
            input_text = f"Department [DEPARTMENT] {level_text} [BUILDING] building."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    <buildingPref building="[BUILDING]" level="{level}"/>
  </department>
</preferences>"""
        
        elif pref_type == "groupPref":
            input_text = f"Department [DEPARTMENT] {level_text} [GROUP] room facilities."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    <groupPref group="[GROUP]" level="{level}"/>
  </department>
</preferences>"""
        
        elif pref_type == "featurePref":
            input_text = f"Department [DEPARTMENT] {level_text} [FEATURE] equipment in classrooms."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    <featurePref feature="[FEATURE]" level="{level}"/>
  </department>
</preferences>"""
        
        elif pref_type == "datePref":
            input_text = f"Department [DEPARTMENT] {level_text} [PATTERN] scheduling pattern."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    <datePref pattern="[PATTERN]" level="{level}"/>
  </department>
</preferences>"""
        
        elif pref_type == "distributionPref":
            input_text = f"Department [DEPARTMENT] {level_text} classes to follow distribution rules."
            dist_xml = generate_distributionPref_xml(level)
            xml = f"""<preferences term="Fal" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    {dist_xml}
  </department>
</preferences>"""
    
    elif element_type == "instructor":
        if pref_type == "timePref":
            input_text = f"Instructor [INSTRUCTOR_NAME] {level_text} teaching on [DAYS] from [START_TIME] to [END_TIME]."
            pref_xml = generate_timePref_xml(level)
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    {pref_xml}
  </instructor>
</preferences>"""
        
        elif pref_type == "coursePref":
            input_text = f"Instructor [INSTRUCTOR_NAME] {level_text} teaching [SUBJECT] [COURSE_ID]."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <coursePref subject="[SUBJECT]" course="[COURSE_ID]" level="{level}"/>
  </instructor>
</preferences>"""
        
        elif pref_type == "teachingPref":
            input_text = f"Instructor [INSTRUCTOR_NAME] {level_text} maximum teaching load of [MAX_LOAD] hours."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <teachingPref maxLoad="[MAX_LOAD]" level="{level}"/>
  </instructor>
</preferences>"""
        
        elif pref_type == "roomPref":
            input_text = f"Instructor [INSTRUCTOR_NAME] {level_text} teaching in room [ROOM_ID]."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <roomPref building="[BUILDING]" room="[ROOM_ID]" level="{level}"/>
  </instructor>
</preferences>"""
        
        elif pref_type == "buildingPref":
            input_text = f"Instructor [INSTRUCTOR_NAME] {level_text} teaching in [BUILDING] building."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <buildingPref building="[BUILDING]" level="{level}"/>
  </instructor>
</preferences>"""
        
        elif pref_type == "groupPref":
            input_text = f"Instructor [INSTRUCTOR_NAME] {level_text} [GROUP] classroom facilities."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <groupPref group="[GROUP]" level="{level}"/>
  </instructor>
</preferences>"""
        
        elif pref_type == "featurePref":
            input_text = f"Instructor [INSTRUCTOR_NAME] {level_text} [FEATURE] equipment for classes."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <featurePref feature="[FEATURE]" level="{level}"/>
  </instructor>
</preferences>"""
        
        elif pref_type == "datePref":
            input_text = f"Instructor [INSTRUCTOR_NAME] {level_text} [PATTERN] class schedule pattern."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <datePref pattern="[PATTERN]" level="{level}"/>
  </instructor>
</preferences>"""
        
        elif pref_type == "distributionPref":
            input_text = f"Instructor [INSTRUCTOR_NAME] {level_text} coordinated class scheduling."
            dist_xml = generate_distributionPref_xml(level)
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    {dist_xml}
  </instructor>
</preferences>"""
    
    elif element_type == "subpart":
        if pref_type == "timePref":
            input_text = f"[SUBJECT] [COURSE_ID] [CLASS_TYPE] {level_text} [DAYS] time slots."
            pref_xml = generate_timePref_xml(level)
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]">
    {pref_xml}
  </subpart>
</preferences>"""
        
        elif pref_type == "roomPref":
            input_text = f"[SUBJECT] [COURSE_ID] [CLASS_TYPE] {level_text} room [ROOM_ID] in [BUILDING]."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]">
    <roomPref building="[BUILDING]" room="[ROOM_ID]" level="{level}"/>
  </subpart>
</preferences>"""
        
        elif pref_type == "buildingPref":
            input_text = f"[SUBJECT] [COURSE_ID] [CLASS_TYPE] {level_text} [BUILDING] building location."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]">
    <buildingPref building="[BUILDING]" level="{level}"/>
  </subpart>
</preferences>"""
        
        elif pref_type == "groupPref":
            input_text = f"[SUBJECT] [COURSE_ID] [CLASS_TYPE] {level_text} [GROUP] facilities."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]">
    <groupPref group="[GROUP]" level="{level}"/>
  </subpart>
</preferences>"""
        
        elif pref_type == "featurePref":
            input_text = f"[SUBJECT] [COURSE_ID] [CLASS_TYPE] {level_text} [FEATURE] equipment."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]">
    <featurePref feature="[FEATURE]" level="{level}"/>
  </subpart>
</preferences>"""
        
        elif pref_type == "datePref":
            input_text = f"[SUBJECT] [COURSE_ID] [CLASS_TYPE] {level_text} [PATTERN] scheduling."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <subpart subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]">
    <datePref pattern="[PATTERN]" level="{level}"/>
  </subpart>
</preferences>"""
    
    else:  # class
        if pref_type == "timePref":
            input_text = f"Class [SUBJECT] [COURSE_ID] section [SUFFIX] {level_text} [DAYS] scheduling."
            pref_xml = generate_timePref_xml(level)
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <class subject="[SUBJECT]" course="[COURSE_ID]" suffix="[SUFFIX]" type="[CLASS_TYPE]">
    {pref_xml}
  </class>
</preferences>"""
        
        elif pref_type == "roomPref":
            input_text = f"Class [SUBJECT] [COURSE_ID] section [SUFFIX] {level_text} room [ROOM_ID]."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <class subject="[SUBJECT]" course="[COURSE_ID]" suffix="[SUFFIX]" type="[CLASS_TYPE]">
    <roomPref building="[BUILDING]" room="[ROOM_ID]" level="{level}"/>
  </class>
</preferences>"""
        
        elif pref_type == "buildingPref":
            input_text = f"Class [SUBJECT] [COURSE_ID] section [SUFFIX] {level_text} [BUILDING] building."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <class subject="[SUBJECT]" course="[COURSE_ID]" suffix="[SUFFIX]" type="[CLASS_TYPE]">
    <buildingPref building="[BUILDING]" level="{level}"/>
  </class>
</preferences>"""
        
        elif pref_type == "groupPref":
            input_text = f"Class [SUBJECT] [COURSE_ID] section [SUFFIX] {level_text} [GROUP] facilities."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <class subject="[SUBJECT]" course="[COURSE_ID]" suffix="[SUFFIX]" type="[CLASS_TYPE]">
    <groupPref group="[GROUP]" level="{level}"/>
  </class>
</preferences>"""
        
        elif pref_type == "featurePref":
            input_text = f"Class [SUBJECT] [COURSE_ID] section [SUFFIX] {level_text} [FEATURE] equipment."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <class subject="[SUBJECT]" course="[COURSE_ID]" suffix="[SUFFIX]" type="[CLASS_TYPE]">
    <featurePref feature="[FEATURE]" level="{level}"/>
  </class>
</preferences>"""
        
        elif pref_type == "datePref":
            input_text = f"Class [SUBJECT] [COURSE_ID] section [SUFFIX] {level_text} [PATTERN] pattern."
            xml = f"""<preferences term="Fall" year="2010" campus="woebegon">
  <class subject="[SUBJECT]" course="[COURSE_ID]" suffix="[SUFFIX]" type="[CLASS_TYPE]">
    <datePref pattern="[PATTERN]" level="{level}"/>
  </class>
</preferences>"""
    
    return {"input": input_text, "output": xml}

def generate_merged_example() -> Dict:
    """Generate examples with multiple elements and preferences"""
    level1, level_text1 = get_random_level_and_text()
    level2, level_text2 = get_random_level_and_text()
    level3, level_text3 = get_random_level_and_text()
    
    scenarios = [
        # Department + Instructor combination
        {
            "input": f"Department [DEPARTMENT] {level_text1} [BUILDING] building and instructor [INSTRUCTOR_NAME] {level_text2} teaching [SUBJECT] courses with [FEATURE] equipment.",
            "xml": f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    <buildingPref building="[BUILDING]" level="{level1}"/>
  </department>
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <coursePref subject="[SUBJECT]" course="[COURSE_ID]" level="{level2}"/>
    <featurePref feature="[FEATURE]" level="{level3}"/>
  </instructor>
</preferences>"""
        },
        
        # Instructor + Class combination
        {
            "input": f"Instructor [INSTRUCTOR_NAME] {level_text1} [GROUP] facilities and class [SUBJECT] [COURSE_ID] {level_text2} room [ROOM_ID].",
            "xml": f"""<preferences term="Fall" year="2010" campus="woebegon">
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <groupPref group="[GROUP]" level="{level1}"/>
  </instructor>
  <class subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]">
    <roomPref building="[BUILDING]" room="[ROOM_ID]" level="{level2}"/>
  </class>
</preferences>"""
        },
        
        # All four elements combination
        {
            "input": f"Department [DEPARTMENT] {level_text1} [BUILDING], instructor [INSTRUCTOR_NAME] {level_text2} morning classes, [SUBJECT] [COURSE_ID] lab {level_text3} computer facilities, and class sections need room assignments.",
            "xml": f"""<preferences term="Fall" year="2010" campus="woebegon">
  <department code="[DEPARTMENT]">
    <buildingPref building="[BUILDING]" level="{level1}"/>
  </department>
  <instructor firstName="[INSTRUCTOR_NAME]" lastName="[LAST_NAME]" department="[DEPARTMENT]">
    <timePref level="{level2}">
      <pref days="[DAYS]" start="0800" stop="1200" level="{level2}"/>
    </timePref>
  </instructor>
  <subpart subject="[SUBJECT]" course="[COURSE_ID]" type="Lab">
    <groupPref group="[GROUP]" level="{level3}"/>
  </subpart>
  <class subject="[SUBJECT]" course="[COURSE_ID]" type="[CLASS_TYPE]">
    <roomPref building="[BUILDING]" room="[ROOM_ID]" level="1"/>
  </class>
</preferences>"""
        }
    ]
    
    return random.choice(scenarios)

def generate_dataset(n_single: int, n_merged: int) -> List[Dict]:
    dataset = [generate_single_intent_example() for _ in range(n_single)] + \
              [generate_merged_example() for _ in range(n_merged)]
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
    dataset = generate_dataset(n_single=150, n_merged=100)
    split_data = split_dataset(dataset)
    save_dataset(split_data)
    print(f"Generated {len(dataset)} examples")
    print(f"Train: {len(split_data['train_pref'])}, Val: {len(split_data['val_pref'])}, Test: {len(split_data['test_pref'])}")

# if __name__ == "__main__":
#     generate_preference_dataset()