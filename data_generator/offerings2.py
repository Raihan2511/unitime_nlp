import json
import random
import os
import re
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from datetime import datetime

class NLPToXMLOfferingsGenerator:
    """
    Generates a dataset of NLP prompts and their corresponding UniTime courseoffering.xml completions.
    The XML format is fixed to match the user's specific target structure.
    """

    ### 1. Initialization and Data Pools ###
    def __init__(self):
        # --- Fixed values to match the target XML format ---
        self.campus = "woebegon"
        self.year = "2010"
        self.term = "Fal"
        self.date_format = "yyyy/M/d"
        self.time_format = "HHmm"
        # --- FIX 4: Hardcode timestamp for consistency ---
        self.created = "Sat Oct 18 19:33:17 CEST 2025"
        self.include_exams = "none"
        self.managing_dept = "0100"
        self.hardcoded_offering_id = "132371" # Hardcoded as requested

        # --- FIX 1: Comprehensive data pools for variability ---
        # Added your new subjects
        self.subjects = [
            "CS", "MATH", "PHYS", "CHEM", "ENGL", "HIST", "BIO", "ECON",
            "DRL", "DLCS", "AOL", "CG", "DBMS", "MVS", "ECN", "MMD"
        ]
        
        # Added "10"
        self.course_numbers = ["10", "101", "106", "201", "301", "350", "401", "499", "220", "330"]
        
        # Added your new subjects and titles
        self.titles = {
            "CS": ["Intro to Programming", "Data Structures", "Algorithms", "Computer Systems"],
            "MATH": ["Calculus I", "Calculus II", "Linear Algebra", "Intro to Statistics"],
            "PHYS": ["General Physics I", "General Physics II", "Modern Mechanics", "Electromagnetism"],
            "CHEM": ["General Chemistry", "Organic Chemistry", "Physical Chemistry", "Biochemistry"],
            "ENGL": ["English Composition", "World Literature", "Creative Writing", "Technical Writing"],
            "HIST": ["World History", "American History", "European History", "Modern History"],
            "BIO": ["Biology I", "Genetics", "Molecular Biology", "Ecology"],
            "ECON": ["Microeconomics", "Macroeconomics", "Economic Theory", "International Economics"],
            # --- New subjects and titles from user request ---
            "DRL": ["Deep Renforcement Learning", "Intro to RL", "RL Agents"],
            "DLCS": ["Deep Learning CyberSecurity", "AI for Security", "Secure Deep Learning", "Deep Learning"],
            "AOL": ["Approximation", "Approximation Algorithms"],
            "CG": ["Computational Geometry", "Geometric Algorithms", "Advanced CG"],
            "DBMS": ["Database Mangament System", "Intro to Databases", "SQL", "NoSQL Databases"],
            "MVS": ["Multivariate Statistics", "Statistical Modeling"],
            "ECN": ["Econometrics", "Intro to Econometrics", "Financial Econometrics"],
            "MMD": ["Massive Data Mining", "Big Data Analytics", "Data Mining II"]
        }
        
        self.buildings = {"Science Hall": "SCI", "Engineering Bldg": "ENGR", "Liberal Arts": "LART", "Education Center": "EDUC"}
        self.room_numbers = ["101", "102", "205", "301", "B08", "Aud1", "205A"]
        
        self.day_patterns = ["MWF", "MW", "TTh", "W", "F", "M", "T"]
        
        self.time_slots = [
            ("0830", "0920", 50), ("0930", "1020", 50), ("1030", "1120", 50),
            ("1330", "1420", 50), ("1430", "1520", 50), ("0900", "1015", 75),
            ("1030", "1145", 75), ("1400", "1515", 75), ("1800", "2050", 170)
        ]
        
        self.limits = [10, 25, 30, 40, 50, 100, 150]
        self.class_types = ["Lec", "Lab", "Rec"]


    ### 2. Helper Methods for Data Generation ###

    def _calculate_min_per_week(self, days: str, duration_minutes: int) -> int:
        """Calculates total instructional minutes per week."""
        day_count = len(days.replace("Th", "R"))
        return day_count * duration_minutes

    def _generate_time_pattern(self, days: str, duration_minutes: int) -> str:
        """Generates a descriptive time pattern string like '3 x 50'."""
        day_count = len(days.replace("Th", "R"))
        return f"{day_count} x {duration_minutes}"
        
    def _generate_course_details(self) -> Dict[str, Any]:
        """Generates a dictionary of structured course data for a single sample."""
        subject = random.choice(self.subjects)
        course_nbr = random.choice(self.course_numbers)
        class_type = random.choice(self.class_types)
        
        start_time, end_time, duration = random.choice(self.time_slots)
        
        # --- FIX 3: Decouple days from duration ---
        # Choose any day pattern. The minPerWeek calculation will handle it.
        days = random.choice(self.day_patterns)

        building_name = random.choice(list(self.buildings.keys()))
        building_code = self.buildings[building_name]
        
        # --- FIX 2a: Randomly decide if a room number should be included ---
        # 20% of the time, we will have no room number.
        if random.random() < 0.20:
            room_nbr = None
        else:
            room_nbr = random.choice(self.room_numbers)
        
        details = {
            "subject": subject,
            "courseNbr": course_nbr,
            "title_desc": random.choice(self.titles[subject]),
            "classType": class_type,
            "limit": random.choice(self.limits),
            "days": days,
            "startTime": start_time,
            "endTime": end_time,
            "duration": duration,
            "buildingName": building_name,
            "buildingCode": building_code,
            "roomNbr": room_nbr, # <-- This can now be None
            "minPerWeek": self._calculate_min_per_week(days, duration),
            "timePattern": self._generate_time_pattern(days, duration)
        }
        return details

    ### 3. NLP and XML Generation Logic ###
    
    def generate_nlp_prompt(self, details: Dict[str, Any]) -> str:
        """Creates a natural language prompt from structured course data."""
        
        # --- FIX 2b: Create two sets of prompt patterns ---
        
        # Patterns for when a room number IS specified
        patterns_with_room = [
            "Please create a new course offering for {subject} {courseNbr}. It's a {classType} section for '{title_desc}', meeting {days} from {startTime} to {endTime} in {buildingName} room {roomNbr}. Set the enrollment limit to {limit} students.",
            "Schedule a {classType} for {subject} {courseNbr}, titled '{title_desc}'. It should run on {days}, {startTime}-{endTime}, in room {roomNbr} of the {buildingName}. The class size is capped at {limit}.",
            "Add a new class: {subject} {courseNbr} ({title_desc}). Place it in {buildingName} {roomNbr} on {days} between {startTime} and {endTime}. It's a {classType} with a limit of {limit}.",
            "I need to set up {subject} {courseNbr}. This is a {classType} called '{title_desc}' meeting {days} at {startTime} until {endTime} in {buildingName} room {roomNbr}. Please cap enrollment at {limit}."
        ]
        
        # Patterns for when a room number IS NOT specified
        patterns_no_room = [
            "Add a new class: {subject} {courseNbr} ({title_desc}). Place it in {buildingName} on {days} between {startTime} and {endTime}. It's a {classType} with a limit of {limit}.",
            "Please schedule {subject} {courseNbr} ({title_desc}) for the {buildingName} building. It's a {classType} on {days}, {startTime} to {endTime}, with a {limit} student cap.",
            "I need a {classType} for {subject} {courseNbr} in {buildingName}. Days: {days}, time: {startTime}-{endTime}, limit: {limit}."
        ]
        
        # Select a pattern list based on whether a room exists
        if details['roomNbr']:
            prompt_template = random.choice(patterns_with_room)
        else:
            prompt_template = random.choice(patterns_no_room)
            
        return prompt_template.format(**details)

    def generate_xml_completion(self, details: Dict[str, Any]) -> str:
        """Generates the target XML from structured course data."""
        
        # --- Derive values that depend on other data points ---
        course_id = str(random.randint(100000, 999999))
        course_title = f"{details['subject']}_{details['courseNbr']}"
        class_suffix = "1" if details['classType'] == "Lec" else details['classType'][0] + "1"
        class_id = f"{details['subject']} {details['courseNbr']} {details['classType']} {class_suffix}"

        # --- FIX 2c: Dynamically create the room tag ---
        if details['roomNbr']:
            room_tag = f'<room building="{details["buildingCode"]}" roomNbr="{details["roomNbr"]}"/>'
        else:
            # Output "TBA" when no room is specified, as discussed
            room_tag = f'<room building="{details["buildingCode"]}" roomNbr="TBA"/>'

        # --- This template is now an exact match of the target format ---
        xml_template = f'''<offerings campus="{self.campus}" year="{self.year}" term="{self.term}" dateFormat="{self.date_format}" timeFormat="{self.time_format}" created="{self.created}" includeExams="{self.include_exams}">
  <offering id="{self.hardcoded_offering_id}" offered="true" action="insert">
    <course id="{course_id}" subject="{details['subject']}" courseNbr="{details['courseNbr']}" controlling="true" title="{course_title}"/>
    <config name="1" limit="{details['limit']}">
      <subpart type="{details['classType']}" suffix="" minPerWeek="{details['minPerWeek']}"/>
      <class id="{class_id}" type="{details['classType']}" suffix="{class_suffix}" limit="{details['limit']}" studentScheduling="true" displayInScheduleBook="true" cancelled="false" managingDept="{self.managing_dept}">
        <time days="{details['days']}" startTime="{details['startTime']}" endTime="{details['endTime']}" timePattern="{details['timePattern']}"/>
        {room_tag}
      </class>
    </config>
  </offering>
</offerings>'''
        
        # Basic formatting to ensure indentation is consistent
        return "\n".join(line.rstrip() for line in xml_template.splitlines())

    ### 4. Main Dataset Creation and Saving Logic ###

    def generate_training_samples(self, count: int) -> List[Dict[str, str]]:
        """
        Generates the specified number of prompt/completion pairs.
        This is the core loop that builds the dataset.
        """
        samples = []
        for _ in range(count):
            # 1. Generate one set of structured course details
            details = self._generate_course_details()
            
            # 2. Create the NLP prompt from these details
            prompt = self.generate_nlp_prompt(details)
            
            # 3. Create the corresponding XML completion from the *same* details
            completion = self.generate_xml_completion(details)
            
            # 4. Store the pair
            samples.append({
                "type": "Course_Offering",
                "prompt": prompt,
                "output": completion
            })
        return samples

    def save_dataset_to_jsonl(self, samples: List[Dict[str, str]], output_dir="data/Courseofferings_dataset"):
        """Splits data and saves it in JSONL format, which is standard for fine-tuning."""
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
    generator = NLPToXMLOfferingsGenerator()
    
    # Generate 2000 samples
    print("Generating 2000 training samples...")
    all_samples = generator.generate_training_samples(count=3000)
    
    # Save the dataset to train/validation/test files
    print("\nSplitting and saving the dataset...")
    generator.save_dataset_to_jsonl(all_samples)
    
    # Display the first 2 generated samples to verify
    print("\n" + "="*80)
    print("Example Generated Samples:")
    print("="*80)
    for i, sample in enumerate(all_samples[:2]):
        print(f"\n--- SAMPLE {i+1} ---")
        print(f"\n[PROMPT]:\n{sample['prompt']}")
        print(f"\n[COMPLETION]:\n{sample['output']}")
    print("\n" + "="*80)