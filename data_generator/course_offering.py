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
        self.term = "Fal"  # Changed from "Fall" to "Fal"
        self.date_format = "yyyy/M/d" # Changed format
        self.time_format = "HHmm"
        # Generate a timestamp that matches the target format for consistency
        self.created = datetime.now().strftime('%a %b %d %H:%M:%S CEST %Y')
        self.include_exams = "none"
        self.managing_dept = "0100"
        self.hardcoded_offering_id = "132371" # Hardcoded as requested

        # --- Comprehensive data pools for variability ---
        self.subjects = ["CS", "MATH", "PHYS", "CHEM", "ENGL", "HIST", "BIO", "ECON"]
        self.course_numbers = ["101", "106", "201", "301", "350", "401", "499"]
        self.titles = {
            "CS": ["Intro to Programming", "Data Structures", "Algorithms", "Computer Systems"],
            "MATH": ["Calculus I", "Calculus II", "Linear Algebra", "Intro to Statistics"],
            "PHYS": ["General Physics I", "General Physics II", "Modern Mechanics", "Electromagnetism"],
            "CHEM": ["General Chemistry", "Organic Chemistry", "Physical Chemistry", "Biochemistry"],
            "ENGL": ["English Composition", "World Literature", "Creative Writing", "Technical Writing"],
            "HIST": ["World History", "American History", "European History", "Modern History"],
            "BIO": ["Biology I", "Genetics", "Molecular Biology", "Ecology"],
            "ECON": ["Microeconomics", "Macroeconomics", "Economic Theory", "International Economics"],
        }
        
        # Using abbreviated building codes to match the example
        self.buildings = {"Science Hall": "SCI", "Engineering Bldg": "ENGR", "Liberal Arts": "LART", "Education Center": "EDUC"}
        self.room_numbers = ["101", "102", "205", "301", "B08", "Aud1", "205A"]
        
        # Day patterns for generation and display
        self.day_patterns = ["MWF", "MW", "TTh", "W", "F"]
        
        # Time slots (start_time_str, end_time_str, duration_minutes)
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
        # A simple mapping for TTh -> 2 days, MWF -> 3 days, etc.
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
        
        # Choose a day pattern appropriate for the duration
        if duration == 50:
            days = "MWF"
        elif duration == 75:
            days = "TTh"
        else:
            days = random.choice(["W", "M", "T"])

        building_name = random.choice(list(self.buildings.keys()))
        building_code = self.buildings[building_name]
        
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
            "roomNbr": random.choice(self.room_numbers),
            "minPerWeek": self._calculate_min_per_week(days, duration),
            "timePattern": self._generate_time_pattern(days, duration)
        }
        return details

    ### 3. NLP and XML Generation Logic ###
    
    def generate_nlp_prompt(self, details: Dict[str, Any]) -> str:
        """Creates a natural language prompt from structured course data."""
        
        # A pool of sentence structures for variety
        patterns = [
            "Please create a new course offering for {subject} {courseNbr}. It's a {classType} section for '{title_desc}', meeting {days} from {startTime} to {endTime} in {buildingName} room {roomNbr}. Set the enrollment limit to {limit} students.",
            "Schedule a {classType} for {subject} {courseNbr}, titled '{title_desc}'. It should run on {days}, {startTime}-{endTime}, in room {roomNbr} of the {buildingName}. The class size is capped at {limit}.",
            "Add a new class: {subject} {courseNbr} ({title_desc}). Place it in {buildingName} {roomNbr} on {days} between {startTime} and {endTime}. It's a {classType} with a limit of {limit}.",
            "I need to set up {subject} {courseNbr}. This is a {classType} called '{title_desc}' meeting {days} at {startTime} until {endTime} in {buildingName} room {roomNbr}. Please cap enrollment at {limit}."
        ]
        
        # Select a random pattern and format it with the details
        prompt_template = random.choice(patterns)
        return prompt_template.format(**details)

    def generate_xml_completion(self, details: Dict[str, Any]) -> str:
        """Generates the target XML from structured course data."""
        
        # --- Derive values that depend on other data points ---
        course_id = str(random.randint(100000, 999999))
        course_title = f"{details['subject']}_{details['courseNbr']}"
        class_suffix = "1" if details['classType'] == "Lec" else details['classType'][0] + "1"
        class_id = f"{details['subject']} {details['courseNbr']} {details['classType']} {class_suffix}"

        # --- This template is now an exact match of the target format ---
        xml_template = f'''<offerings campus="{self.campus}" year="{self.year}" term="{self.term}" dateFormat="{self.date_format}" timeFormat="{self.time_format}" created="{self.created}" includeExams="{self.include_exams}">
  <offering id="{self.hardcoded_offering_id}" offered="true" action="insert">
    <course id="{course_id}" subject="{details['subject']}" courseNbr="{details['courseNbr']}" controlling="true" title="{course_title}"/>
    <config name="1" limit="{details['limit']}">
      <subpart type="{details['classType']}" suffix="" minPerWeek="{details['minPerWeek']}"/>
      <class id="{class_id}" type="{details['classType']}" suffix="{class_suffix}" limit="{details['limit']}" studentScheduling="true" displayInScheduleBook="true" cancelled="false" managingDept="{self.managing_dept}">
        <time days="{details['days']}" startTime="{details['startTime']}" endTime="{details['endTime']}" timePattern="{details['timePattern']}"/>
        <room building="{details['buildingCode']}" roomNbr="{details['roomNbr']}"/>
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
                "context": "COURSE OFFERING REQUEST",
                "prompt": prompt,
                "output": completion
            })
        return samples

    def save_dataset_to_jsonl(self, samples: List[Dict[str, str]], output_dir="/home/sysadm/Music/unitime_nlp/data/Courseofferings_dataset"):
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
    
    # Generate 1000 samples
    print("Generating 1000 training samples...")
    all_samples = generator.generate_training_samples(count=1000)
    
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