import json
import random
import os
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

class NLPToXMLOfferingsGenerator:
    def __init__(self):
        # Fixed values
        self.campus = "webegon"
        self.year = "2010"
        self.term = "Fall"
        self.date_format = "MM/dd/yyyy"
        self.created = "06/22/2010"
        
        # Data pools for replacement
        self.subjects = ["CS", "MATH", "PHYS", "CHEM", "ENGL", "HIST", "BIO", "ECON", "PHIL", "STAT"]
        self.course_numbers = ["101", "102", "201", "301", "350", "401", "450", "499"]
        self.titles = [
            "Introduction to Programming", "Calculus I", "Physics Fundamentals", 
            "Chemistry Basics", "English Composition", "World History", 
            "Biology Principles", "Economics Theory", "Philosophy Ethics", "Statistics Methods"
        ]
        self.credits = ["3", "4", "5"]
        self.limits = ["25", "30", "35", "40", "50", "75", "100"]
        self.buildings = ["Science Hall", "Engineering Bldg", "Liberal Arts", "Business Center", "Tech Center"]
        self.room_numbers = ["101", "102", "205", "301", "Lab1", "Aud100"]
        self.days_patterns = ["MWF", "TTH", "MW", "TR", "MTWF"]
        self.start_times = ["0800", "0900", "1000", "1100", "1300", "1400", "1500"]
        self.end_times = ["0950", "1050", "1150", "1250", "1450", "1550", "1650"]
        self.first_names = ["John", "Sarah", "Mike", "Lisa", "David", "Anna", "Chris", "Maria"]
        self.last_names = ["Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"]
        
        # Conversational starters
        self.conversation_starters = [
            "Create course", "Schedule", "Set up", "Add course", "I need to create",
            "Can you schedule", "Please add", "Set up a new course", "Create a class for"
        ]
    
    def replace_placeholders(self, text: str) -> str:
        """Replace placeholders with actual values"""
        # Generate values that depend on each other first
        subject_1 = random.choice(self.subjects)
        subject_2 = random.choice([s for s in self.subjects if s != subject_1])
        
        building_1 = random.choice(self.buildings)
        building_2 = random.choice([b for b in self.buildings if b != building_1])
        
        limit = random.choice(self.limits)
        lab_limit = str(int(limit) // 2)
        rec_limit = str(int(limit) // 2)
        dis_limit = str(int(limit) // 2)
        section_limit = str(int(limit) // 2)
        
        replacements = {
            "[SUBJECT]": random.choice(self.subjects),
            "[SUBJECT_1]": subject_1,
            "[SUBJECT_2]": subject_2,
            "[COURSE_NBR]": random.choice(self.course_numbers),
            "[COURSE_NBR_1]": random.choice(self.course_numbers),
            "[COURSE_NBR_2]": random.choice(self.course_numbers),
            "[TITLE]": random.choice(self.titles),
            "[CREDITS]": random.choice(self.credits),
            "[LIMIT]": limit,
            "[LAB_LIMIT]": lab_limit,
            "[REC_LIMIT]": rec_limit,
            "[DIS_LIMIT]": dis_limit,
            "[SECTION_LIMIT]": section_limit,
            "[TOTAL_LIMIT]": random.choice(self.limits),
            "[BUILDING]": random.choice(self.buildings),
            "[BUILDING_1]": building_1,
            "[BUILDING_2]": building_2,
            "[ROOM_NBR]": random.choice(self.room_numbers),
            "[ROOM_NBR_1]": random.choice(self.room_numbers),
            "[ROOM_NBR_2]": random.choice(self.room_numbers),
            "[DAYS]": random.choice(self.days_patterns),
            "[DAYS_1]": random.choice(self.days_patterns),
            "[DAYS_2]": random.choice(self.days_patterns),
            "[START_TIME]": random.choice(self.start_times),
            "[START_TIME_1]": random.choice(self.start_times),
            "[START_TIME_2]": random.choice(self.start_times),
            "[END_TIME]": random.choice(self.end_times),
            "[END_TIME_1]": random.choice(self.end_times),
            "[END_TIME_2]": random.choice(self.end_times),
            "[INSTRUCTOR_FNAME]": random.choice(self.first_names),
            "[INSTRUCTOR_FNAME_1]": random.choice(self.first_names),
            "[INSTRUCTOR_FNAME_2]": random.choice(self.first_names),
            "[INSTRUCTOR_LNAME]": random.choice(self.last_names),
            "[INSTRUCTOR_LNAME_1]": random.choice(self.last_names),
            "[INSTRUCTOR_LNAME_2]": random.choice(self.last_names),
            "[OFFERING_ID]": str(random.randint(1000, 9999)),
            "[CLASS_ID]": str(random.randint(100, 999)),
            "[LEC_CLASS_ID]": str(random.randint(100, 999)),
            "[LAB_CLASS_ID]": str(random.randint(100, 999)),
            "[REC_CLASS_ID]": str(random.randint(100, 999)),
            "[DIS_CLASS_ID]": str(random.randint(100, 999)),
            "[CLASS_ID_1]": str(random.randint(100, 999)),
            "[CLASS_ID_2]": str(random.randint(100, 999)),
            "[INSTRUCTOR_ID]": str(random.randint(1000, 9999)),
            "[INSTRUCTOR_ID_1]": str(random.randint(1000, 9999)),
            "[INSTRUCTOR_ID_2]": str(random.randint(1000, 9999)),
            "[MIN_PER_WEEK]": str(random.choice([150, 180, 210])),
            "[LEC_MIN_PER_WEEK]": str(random.choice([150, 180])),
            "[LAB_MIN_PER_WEEK]": str(random.choice([120, 150])),
            "[REC_MIN_PER_WEEK]": str(random.choice([50, 75])),
            "[DIS_MIN_PER_WEEK]": str(random.choice([50, 75])),
            "[NUM_SECTIONS]": str(random.choice([2, 3])),
            "[SHARE_1]": str(random.choice([60, 70])),
            "[SHARE_2]": str(random.choice([30, 40])),
            "[EXAM_ID]": str(random.randint(500, 999)),
            "[EXAM_NAME]": "Final Exam",
            "[EXAM_LENGTH]": str(random.choice([120, 150, 180])),
            "[EXAM_DATE]": "12/15/2010",
            "[EXAM_START_TIME]": "0800",
            "[EXAM_END_TIME]": "1000",
            "[EXAM_BUILDING]": random.choice(self.buildings),
            "[EXAM_ROOM_NBR]": random.choice(self.room_numbers)
        }
        
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        return text
        """Replace placeholders with actual values"""    
    def make_conversational(self, template: str) -> str:
        """Make the NLP more conversational"""
        starter = random.choice(self.conversation_starters)
        
        # Remove "Create course" from beginning if it exists and add our starter
        if template.startswith("Create course"):
            template = template[13:].strip()
        elif template.startswith("Schedule"):
            template = template[8:].strip()
        elif template.startswith("Course"):
            template = template[6:].strip()
        elif template.startswith("Cross-listed course"):
            template = template[19:].strip()
        elif template.startswith("Team taught course"):
            template = template[18:].strip()
        elif template.startswith("Complex course"):
            template = template[14:].strip()
            
        return f"{starter} {template}"
    
    def generate_offerings_samples(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate course offerings NLP to XML samples with actual values"""
        patterns = [
            {
                "nlp_template": "Create course [SUBJECT] [COURSE_NBR] titled [TITLE] with [CREDITS] credit hours lecture only limit [LIMIT] students",
                "xml_template": f"""<?xml version="1.0" ?>
<offerings campus="{self.campus}" year="{self.year}" term="{self.term}" dateFormat="{self.date_format}" created="{self.created}">
  <offering id="[OFFERING_ID]" offered="true">
    <courseCredit creditType="collegiate" creditUnitType="semesterHours" creditFormat="fixedUnit" fixedCredit="[CREDITS]"/>
    <course subject="[SUBJECT]" courseNbr="[COURSE_NBR]" controlling="true" title="[TITLE]"/>
    <config name="1" limit="[LIMIT]">
      <subpart type="Lec" minPerWeek="[MIN_PER_WEEK]"/>
      <class id="[CLASS_ID]" type="Lec" suffix="1" limit="[LIMIT]">
        <time days="[DAYS]" startTime="[START_TIME]" endTime="[END_TIME]"/>
        <room building="[BUILDING]" roomNbr="[ROOM_NBR]"/>
        <instructor id="[INSTRUCTOR_ID]" fname="[INSTRUCTOR_FNAME]" lname="[INSTRUCTOR_LNAME]" lead="true"/>
      </class>
    </config>
  </offering>
</offerings>"""
            },
            {
                "nlp_template": "[SUBJECT] [COURSE_NBR] [TITLE] lecture [DAYS] [START_TIME] to [END_TIME] in [BUILDING] room [ROOM_NBR] with Professor [INSTRUCTOR_LNAME] [CREDITS] credits limit [LIMIT]",
                "xml_template": f"""<?xml version="1.0" ?>
<offerings campus="{self.campus}" year="{self.year}" term="{self.term}" dateFormat="{self.date_format}" created="{self.created}">
  <offering id="[OFFERING_ID]" offered="true">
    <courseCredit creditType="collegiate" creditUnitType="semesterHours" creditFormat="fixedUnit" fixedCredit="[CREDITS]"/>
    <course subject="[SUBJECT]" courseNbr="[COURSE_NBR]" controlling="true" title="[TITLE]"/>
    <config name="1" limit="[LIMIT]">
      <subpart type="Lec" minPerWeek="[MIN_PER_WEEK]"/>
      <class id="[CLASS_ID]" type="Lec" suffix="1" limit="[LIMIT]">
        <time days="[DAYS]" startTime="[START_TIME]" endTime="[END_TIME]"/>
        <room building="[BUILDING]" roomNbr="[ROOM_NBR]"/>
        <instructor id="[INSTRUCTOR_ID]" fname="[INSTRUCTOR_FNAME]" lname="[INSTRUCTOR_LNAME]" lead="true"/>
      </class>
    </config>
  </offering>
</offerings>"""
            },
            {
                "nlp_template": "Schedule [SUBJECT] [COURSE_NBR] [TITLE] with lecture and lab [CREDITS] credits limit [LIMIT] students",
                "xml_template": f"""<?xml version="1.0" ?>
<offerings campus="{self.campus}" year="{self.year}" term="{self.term}" dateFormat="{self.date_format}" created="{self.created}">
  <offering id="[OFFERING_ID]" offered="true">
    <courseCredit creditType="collegiate" creditUnitType="semesterHours" creditFormat="fixedUnit" fixedCredit="[CREDITS]"/>
    <course subject="[SUBJECT]" courseNbr="[COURSE_NBR]" controlling="true" title="[TITLE]"/>
    <config name="1" limit="[LIMIT]">
      <subpart type="Lec" minPerWeek="[LEC_MIN_PER_WEEK]">
        <subpart type="Lab" minPerWeek="[LAB_MIN_PER_WEEK]"/>
      </subpart>
      <class id="[LEC_CLASS_ID]" type="Lec" suffix="1" limit="[LIMIT]">
        <class id="[LAB_CLASS_ID]" type="Lab" suffix="1" limit="[LAB_LIMIT]"/>
        <time days="[DAYS]" startTime="[START_TIME]" endTime="[END_TIME]"/>
        <room building="[BUILDING]" roomNbr="[ROOM_NBR]"/>
        <instructor id="[INSTRUCTOR_ID]" fname="[INSTRUCTOR_FNAME]" lname="[INSTRUCTOR_LNAME]" lead="true"/>
      </class>
    </config>
  </offering>
</offerings>"""
            }
        ]
        
        samples = []
        # Generate multiple variations of each pattern
        for _ in range(count):
            pattern = random.choice(patterns)
            
            # Replace placeholders with actual values
            nlp_text = self.replace_placeholders(pattern["nlp_template"])
            xml_text = self.replace_placeholders(pattern["xml_template"])
            
            # Make NLP more conversational
            nlp_text = self.make_conversational(nlp_text)
            
            samples.append({
                "input": nlp_text,
                "output": xml_text
            })
        
        return samples
    
    def split_and_save_dataset(self, samples: List[Dict[str, Any]], output_dir="data/offerings_data"):
        os.makedirs(output_dir, exist_ok=True)
        
        # Split into train/val/test
        train, temp = train_test_split(samples, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        splits = {
            "train_offer.json": train,
            "val_offer.json": val,
            "test_offer.json": test
        }
        
        for fname, data in splits.items():
            path = os.path.join(output_dir, fname)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(data)} samples to {path}")
    
    def generate_and_save(self, count: int = 100):
        """Generate, split, and save the dataset"""
        samples = self.generate_offerings_samples(count)
        self.split_and_save_dataset(samples)
        return samples

# Usage
def generate_offerings_dataset():
    generator = NLPToXMLOfferingsGenerator()
    samples = generator.generate_and_save(1000)  # Generate 100 samples
    
    print("Sample conversations:")
    print("=" * 80)
    for i, sample in enumerate(samples[:3]):
        print(f"\nExample {i+1}:")
        print(f"User: {sample['input']}")
        print(f"XML: {sample['output'][:300]}...")
        print("-" * 40)

# if __name__ == "__main__":
#     generate_offerings_dataset()