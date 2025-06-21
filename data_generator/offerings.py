import json
import random
import os
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

class NLPToXMLOfferingsGenerator:
    def __init__(self):
        # Fixed values - same for all samples
        self.campus = "PuWL"
        self.year = "2025"
        self.term = "Fall"
        self.date_format = "MM/dd/yyyy"
        self.created = "06/22/2025"
        
    def generate_offerings_samples(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate course offerings NLP to XML samples with placeholders in NLP too"""
        samples = []
        
        # Template patterns - placeholders in BOTH NLP and XML
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
                "nlp_template": "Schedule [SUBJECT] [COURSE_NBR] [TITLE] with lecture and lab components [CREDITS] credits limit [LIMIT] students lab limit [LAB_LIMIT]",
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
            },
            {
                "nlp_template": "Course [SUBJECT] [COURSE_NBR] [TITLE] lecture [DAYS] [START_TIME] to [END_TIME] in [BUILDING] room [ROOM_NBR] with instructor [INSTRUCTOR_LNAME] [CREDITS] credits limit [LIMIT]",
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
                "nlp_template": "[SUBJECT] [COURSE_NBR] [TITLE] with [NUM_SECTIONS] lecture sections [CREDITS] credit hours each section limited to [SECTION_LIMIT] students total limit [TOTAL_LIMIT]",
                "xml_template": f"""<?xml version="1.0" ?>
<offerings campus="{self.campus}" year="{self.year}" term="{self.term}" dateFormat="{self.date_format}" created="{self.created}">
  <offering id="[OFFERING_ID]" offered="true">
    <courseCredit creditType="collegiate" creditUnitType="semesterHours" creditFormat="fixedUnit" fixedCredit="[CREDITS]"/>
    <course subject="[SUBJECT]" courseNbr="[COURSE_NBR]" controlling="true" title="[TITLE]"/>
    <config name="1" limit="[TOTAL_LIMIT]">
      <subpart type="Lec" minPerWeek="[MIN_PER_WEEK]"/>
      <class id="[CLASS_ID_1]" type="Lec" suffix="1" limit="[SECTION_LIMIT]">
        <time days="[DAYS_1]" startTime="[START_TIME_1]" endTime="[END_TIME_1]"/>
        <room building="[BUILDING_1]" roomNbr="[ROOM_NBR_1]"/>
        <instructor id="[INSTRUCTOR_ID_1]" fname="[INSTRUCTOR_FNAME_1]" lname="[INSTRUCTOR_LNAME_1]" lead="true"/>
      </class>
      <class id="[CLASS_ID_2]" type="Lec" suffix="2" limit="[SECTION_LIMIT]">
        <time days="[DAYS_2]" startTime="[START_TIME_2]" endTime="[END_TIME_2]"/>
        <room building="[BUILDING_2]" roomNbr="[ROOM_NBR_2]"/>
        <instructor id="[INSTRUCTOR_ID_2]" fname="[INSTRUCTOR_FNAME_2]" lname="[INSTRUCTOR_LNAME_2]" lead="true"/>
      </class>
    </config>
  </offering>
</offerings>"""
            },
            {
                "nlp_template": "Complex course [SUBJECT] [COURSE_NBR] [TITLE] with lecture recitation and lab components [CREDITS] credits limit [LIMIT] recitation limit [REC_LIMIT] lab limit [LAB_LIMIT] with final exam [EXAM_LENGTH] minutes",
                "xml_template": f"""<?xml version="1.0" ?>
<offerings campus="{self.campus}" year="{self.year}" term="{self.term}" dateFormat="{self.date_format}" created="{self.created}">
  <offering id="[OFFERING_ID]" offered="true">
    <courseCredit creditType="collegiate" creditUnitType="semesterHours" creditFormat="fixedUnit" fixedCredit="[CREDITS]"/>
    <course subject="[SUBJECT]" courseNbr="[COURSE_NBR]" controlling="true" title="[TITLE]"/>
    <config name="1" limit="[LIMIT]">
      <subpart type="Lec" minPerWeek="[LEC_MIN_PER_WEEK]">
        <subpart type="Rec" minPerWeek="[REC_MIN_PER_WEEK]"/>
        <subpart type="Lab" minPerWeek="[LAB_MIN_PER_WEEK]"/>
      </subpart>
      <class id="[LEC_CLASS_ID]" type="Lec" suffix="1" limit="[LIMIT]">
        <class id="[REC_CLASS_ID]" type="Rec" suffix="1" limit="[REC_LIMIT]"/>
        <class id="[LAB_CLASS_ID]" type="Lab" suffix="1" limit="[LAB_LIMIT]"/>
        <time days="[DAYS]" startTime="[START_TIME]" endTime="[END_TIME]"/>
        <room building="[BUILDING]" roomNbr="[ROOM_NBR]"/>
        <instructor id="[INSTRUCTOR_ID]" fname="[INSTRUCTOR_FNAME]" lname="[INSTRUCTOR_LNAME]" lead="true"/>
      </class>
    </config>
    <exam id="[EXAM_ID]" name="[EXAM_NAME]" length="[EXAM_LENGTH]" type="final">
      <period date="[EXAM_DATE]" startTime="[EXAM_START_TIME]" endTime="[EXAM_END_TIME]"/>
      <room building="[EXAM_BUILDING]" roomNbr="[EXAM_ROOM_NBR]"/>
    </exam>
  </offering>
</offerings>"""
            },
            {
                "nlp_template": "Cross-listed course [SUBJECT_1] [COURSE_NBR_1] and [SUBJECT_2] [COURSE_NBR_2] titled [TITLE] lecture and discussion [CREDITS] credits limit [LIMIT] discussion limit [DIS_LIMIT]",
                "xml_template": f"""<?xml version="1.0" ?>
<offerings campus="{self.campus}" year="{self.year}" term="{self.term}" dateFormat="{self.date_format}" created="{self.created}">
  <offering id="[OFFERING_ID]" offered="true">
    <courseCredit creditType="collegiate" creditUnitType="semesterHours" creditFormat="fixedUnit" fixedCredit="[CREDITS]"/>
    <course subject="[SUBJECT_1]" courseNbr="[COURSE_NBR_1]" controlling="true" title="[TITLE]"/>
    <course subject="[SUBJECT_2]" courseNbr="[COURSE_NBR_2]" controlling="false" title="[TITLE]"/>
    <config name="1" limit="[LIMIT]">
      <subpart type="Lec" minPerWeek="[LEC_MIN_PER_WEEK]">
        <subpart type="Dis" minPerWeek="[DIS_MIN_PER_WEEK]"/>
      </subpart>
      <class id="[LEC_CLASS_ID]" type="Lec" suffix="1" limit="[LIMIT]">
        <class id="[DIS_CLASS_ID]" type="Dis" suffix="1" limit="[DIS_LIMIT]"/>
        <time days="[DAYS]" startTime="[START_TIME]" endTime="[END_TIME]"/>
        <room building="[BUILDING]" roomNbr="[ROOM_NBR]"/>
        <instructor id="[INSTRUCTOR_ID]" fname="[INSTRUCTOR_FNAME]" lname="[INSTRUCTOR_LNAME]" lead="true"/>
      </class>
    </config>
  </offering>
</offerings>"""
            },
            {
                "nlp_template": "Team taught course [SUBJECT] [COURSE_NBR] [TITLE] with instructor [INSTRUCTOR_LNAME_1] share [SHARE_1] and instructor [INSTRUCTOR_LNAME_2] share [SHARE_2] total [CREDITS] credits limit [LIMIT]",
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
        <instructor id="[INSTRUCTOR_ID_1]" fname="[INSTRUCTOR_FNAME_1]" lname="[INSTRUCTOR_LNAME_1]" share="[SHARE_1]" lead="true"/>
        <instructor id="[INSTRUCTOR_ID_2]" fname="[INSTRUCTOR_FNAME_2]" lname="[INSTRUCTOR_LNAME_2]" share="[SHARE_2]"/>
      </class>
    </config>
  </offering>
</offerings>"""
            }
        ]
        
        # Generate samples - just return the templates as-is with placeholders
        for pattern in patterns:
            samples.append({
                "nlp_input": pattern["nlp_template"],
                "xml_output": pattern["xml_template"]
            })
        
        return samples
    
    # def save_dataset(self, samples: List[Dict[str, Any]], filename: str = "nlp_xml_offerings_dataset.json"):
    #     """Save the generated dataset to a JSON file"""
    #     with open(filename, 'w', encoding='utf-8') as f:
    #         json.dump(samples, f, indent=2, ensure_ascii=False)
    #     print(f"Dataset saved to {filename} with {len(samples)} samples")
    
    # def generate_and_save(self):
    #     """Generate and save the complete dataset"""
    #     samples = self.generate_offerings_samples()
    #     self.save_dataset(samples)
    #     return samples
    def split_and_save_dataset(self, samples: List[Dict[str, Any]], output_dir="data/offerings_data"):
        os.makedirs(output_dir, exist_ok=True)

        # Split into train/val/test
        train, temp = train_test_split(samples, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        # Save them
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

    def generate_and_save(self):
        """Generate, split, and save the dataset"""
        samples = self.generate_offerings_samples()
        self.split_and_save_dataset(samples)
        return samples


# Example usage
def generate_offerings_dataset():

    generator = NLPToXMLOfferingsGenerator()
    
    # Generate sample data
    samples = generator.generate_and_save()
    
    # Print all samples
    print("NLP-to-XML Template Pairs:")
    print("=" * 80)
    for i, sample in enumerate(samples):
        print(f"\nTemplate {i+1}:")
        print(f"NLP: {sample['nlp_input']}")
        print(f"XML: {sample['xml_output'][:300]}...")
        print("-" * 80)

# if __name__ == "__main__":
#     generate_offerings_dataset()
