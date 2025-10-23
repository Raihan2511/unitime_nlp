import json
import random
import os
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

class NLPToXMLPreferencesGenerator:
    """
    Generates a dataset of NLP prompts and their corresponding UniTime preferences.xml completions.
    This script creates samples for multiple preference types (instructor, department, subpart).
    """

    ### 1. Initialization and Data Pools ###
    def __init__(self):
        # --- Fixed values from the target XML format ---
        self.campus = "woebegon"
        self.year = "2010"
        self.term = "Fal"
        self.date_format = "yyyy/M/d"
        self.time_format = "HHmm"
        # --- Hardcoded timestamp for consistency ---
        self.created = "Thu Oct 23 21:38:17 CEST 2025" 

        # --- Data Pools ---
        self.pref_levels = {
            "P": "preferred",
            "2": "strongly preferred",
            "1": "weakly preferred",
            "-1": "discouraged",
            "-2": "prohibited",
            "R": "required"
        }
        
        # Departments
        self.dept_codes = ["0100", "0101", "0102"]
        
        # Rooms (Building + Number)
        self.rooms = ["EDUC 101", "THTR 101", "EDUC 106", "EDUC 102", "MALL", "SCI 205"]
        
        # Instructors
        self.instructors = [
            ("100", "JOE", "DOE", "0101"),
            ("101", "GEORGE", "NEWMAN", "0101"),
            ("102", "JOHN", "SMITH", "0101"),
            ("103", "SARAH", "JENKINS", "0102")
        ]
        
        # Room Features
        self.features = ["Comp", "ThtrSeat", "Projector", "Whiteboard"]
        
        # Buildings
        self.buildings = ["EDUC", "THTR", "MALL", "SCI", "LART"]
        
        # Days (R = Thursday)
        self.days = ["M", "T", "W", "R", "F"]
        
        # Time Slots (start, stop)
        self.time_slots = [
            ("0730", "0830"), ("0930", "1430"), ("0830", "1830"),
            ("0930", "1330"), ("1630", "1830")
        ]
        
        # Subparts (Courses)
        self.subparts = [
            ("ALG", "101", "Lec"), ("BIOL", "101", "Lab"), ("BIOL", "101", "Lec"),
            ("C S", "101", "Lab"), ("CHM", "101", "Lec"), ("ECON", "101", "Lec")
        ]
        
        # Time Patterns
        self.patterns = ["3 x 50", "2 x 75", "1 x 100", "1 x 150", "5 x 100"]
        
        # Room Groups
        self.groups = ["Classroom", "Comp Labs", "Biol Labs"]

        # --- List of all generator functions to call ---
        self.generators = [
            self.generate_dept_room_pref,
            self.generate_instructor_time_pref,
            self.generate_instructor_feature_pref,
            self.generate_instructor_building_pref,
            self.generate_subpart_group_pref,
            self.generate_subpart_time_pref
        ]

    ### 2. Helper Methods ###

    def _get_random_level(self) -> Tuple[str, str]:
        """Picks a random preference level code and its name."""
        level_code = random.choice(list(self.pref_levels.keys()))
        level_name = self.pref_levels[level_code]
        return level_code, level_name

    def _wrap_xml(self, content: str) -> str:
        """Wraps the generated XML snippet in the root <preferences> tag."""
        xml = f'''<preferences term="{self.term}" year="{self.year}" campus="{self.campus}" dateFormat="{self.date_format}" timeFormat="{self.time_format}" created="{self.created}">
{content}
</preferences>'''
        return "\n".join(line.rstrip() for line in xml.splitlines())

    ### 3. Specific Preference Generators ###

    def generate_dept_room_pref(self) -> Dict[str, str]:
        """Generates a sample for a department's room preference."""
        dept = random.choice(self.dept_codes)
        room = random.choice(self.rooms)
        level_code, level_name = self._get_random_level()

        prompt = f"Set a room preference for department {dept}: room {room} should be {level_name}."
        
        xml_content = f'''<department code="{dept}">
<roomPref location="{room}" level="{level_code}"/>
</department>'''
        
        return {
            "type": "DepartmentRoomPref",
            "prompt": prompt,
            "output": self._wrap_xml(xml_content)
        }

    def generate_instructor_time_pref(self) -> Dict[str, str]:
        """Generates a sample for an instructor's time preference."""
        inst_id, f_name, l_name, dept = random.choice(self.instructors)
        
        # Create a "main" preference for the prompt
        main_day = random.choice(self.days)
        main_start, main_stop = random.choice(self.time_slots)
        main_level, main_level_name = self._get_random_level()
        
        prompt = f"For instructor {f_name} {l_name}, make the time slot {main_day} {main_start}-{main_stop} {main_level_name}."

        # Generate a list of time preferences to make the XML realistic
        prefs_xml = ""
        # Add the main preference
        prefs_xml += f'<pref level="{main_level}" day="{main_day}" start="{main_start}" stop="{main_stop}"/>\n'
        
        # Add a few other random ones for boilerplate
        for _ in range(random.randint(2, 4)):
            day = random.choice(self.days)
            start, stop = random.choice(self.time_slots)
            level, _ = self._get_random_level()
            prefs_xml += f'<pref level="{level}" day="{day}" start="{start}" stop="{stop}"/>\n'

        xml_content = f'''<instructor externalId="{inst_id}" firstName="{f_name}" lastName="{l_name}" department="{dept}">
<timePref level="R">
{prefs_xml.strip()}
</timePref>
</instructor>'''
        
        return {
            "type": "InstructorTimePref",
            "prompt": prompt,
            "output": self._wrap_xml(xml_content)
        }

    def generate_instructor_feature_pref(self) -> Dict[str, str]:
        """Generates a sample for an instructor's room feature preference."""
        inst_id, f_name, l_name, dept = random.choice(self.instructors)
        feature = random.choice(self.features)
        level_code, level_name = self._get_random_level()

        prompt = f"Instructor {f_name} {l_name} has a {level_name} preference for the '{feature}' room feature."
        
        xml_content = f'''<instructor externalId="{inst_id}" firstName="{f_name}" lastName="{l_name}" department="{dept}">
<featurePref feature="{feature}" level="{level_code}"/>
</instructor>'''
        
        return {
            "type": "InstructorFeaturePref",
            "prompt": prompt,
            "output": self._wrap_xml(xml_content)
        }

    def generate_instructor_building_pref(self) -> Dict[str, str]:
        """Generates a sample for an instructor's building preference."""
        inst_id, f_name, l_name, dept = random.choice(self.instructors)
        building = random.choice(self.buildings)
        level_code, level_name = self._get_random_level()

        prompt = f"Set a building preference for {f_name} {l_name}: {building} is {level_name}."
        
        xml_content = f'''<instructor externalId="{inst_id}" firstName="{f_name}" lastName="{l_name}" department="{dept}">
<buildingPref building="{building}" level="{level_code}"/>
</instructor>'''
        
        return {
            "type": "InstructorBuildingPref",
            "prompt": prompt,
            "output": self._wrap_xml(xml_content)
        }

    def generate_subpart_group_pref(self) -> Dict[str, str]:
        """Generates a sample for a course subpart's room group preference."""
        subject, course, type = random.choice(self.subparts)
        group = random.choice(self.groups)
        level_code, level_name = self._get_random_level()
        
        prompt = f"The {subject} {course} {type} course subpart has a {level_name} preference for the '{group}' room group."
        
        xml_content = f'''<subpart subject="{subject}" course="{course}" type="{type}">
<groupPref group="{group}" level="{level_code}"/>
</subpart>'''
        
        return {
            "type": "SubpartGroupPref",
            "prompt": prompt,
            "output": self._wrap_xml(xml_content)
        }

    def generate_subpart_time_pref(self) -> Dict[str, str]:
        """Generates a sample for a course subpart's time pattern preference."""
        subject, course, type = random.choice(self.subparts)
        pattern = random.choice(self.patterns)
        level_code, level_name = self._get_random_level()

        # Decide if we're adding specific times or just the pattern
        if random.random() < 0.5:
            # --- Simple prompt: Just the pattern ---
            prompt = f"For the {subject} {course} {type}, the time pattern {pattern} is {level_name}."
            
            xml_content = f'''<subpart subject="{subject}" course="{course}" type="{type}">
<timePref pattern="{pattern}" level="{level_code}"/>
</subpart>'''
            
        else:
            # --- Complex prompt: Pattern + specific times ---
            day = random.choice(["M", "T", "W", "R", "F", "TR", "MW"])
            time = random.choice(["0730", "0900", "1030", "1330"])
            prompt = f"For {subject} {course} {type}, the {pattern} pattern is {level_name}, especially {day} at {time}."
            
            # The inner <pref> level is usually 'P' in the example
            xml_content = f'''<subpart subject="{subject}" course="{course}" type="{type}">
<timePref pattern="{pattern}" level="{level_code}">
<pref level="P" days="{day}" time="{time}"/>
</timePref>
</subpart>'''

        return {
            "type": "SubpartTimePref",
            "prompt": prompt,
            "output": self._wrap_xml(xml_content)
        }

    ### 4. Main Dataset Creation and Saving Logic ###

    def generate_training_samples(self, count: int) -> List[Dict[str, str]]:
        """
        Generates the specified number of prompt/completion pairs by
        randomly calling the different generator functions.
        """
        samples = []
        for _ in range(count):
            # 1. Randomly pick one of the generator functions
            generator_func = random.choice(self.generators)
            
            # 2. Call it to get a complete sample
            sample = generator_func()
            
            # 3. Store the pair
            samples.append(sample)
        return samples

    def save_dataset_to_jsonl(self, samples: List[Dict[str, str]], output_dir="data/Preferences_dataset"):
        """Splits data and saves it in JSONL format."""
        os.makedirs(output_dir, exist_ok=True)
        
        train, temp = train_test_split(samples, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        splits = {"train.jsonl": train, "validation.jsonl": val, "test.jsonl": test}
        
        for filename, data in splits.items():
            path = os.path.join(output_dir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                for entry in data:
                    json.dump(entry, f)
                    f.write('\n')
            print(f"Saved {len(data)} samples to {path}")

# --- Execution ---
if __name__ == "__main__":
    # Initialize the generator
    generator = NLPToXMLPreferencesGenerator()
    
    # Generate 2000 samples
    print("Generating 2000 training samples...")
    all_samples = generator.generate_training_samples(count=2000)
    
    # Save the dataset to train/validation/test files
    print("\nSplitting and saving the dataset...")
    generator.save_dataset_to_jsonl(all_samples)
    
    # Display the first 5 generated samples to verify
    print("\n" + "="*80)
    print("Example Generated Samples (showing 5 random types):")
    print("="*80)
    for i, sample in enumerate(all_samples[:5]):
        print(f"\n--- SAMPLE {i+1} (Type: {sample['type']}) ---")
        print(f"\n[PROMPT]:\n{sample['prompt']}")
        print(f"\n[COMPLETION]:\n{sample['output']}")
    print("\n" + "="*80)