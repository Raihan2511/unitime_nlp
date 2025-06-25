import json
import random
import os
import re
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

class NLPToXMLOfferingsGenerator:
    def __init__(self):
        # Fixed values for consistency
        self.campus = "webegon"
        self.year = "2010"
        self.term = "Fall"
        self.date_format = "MM/dd/yyyy"
        self.created = "06/22/2010"
        
        # Comprehensive data pools
        self.subjects = ["CS", "MATH", "PHYS", "CHEM", "ENGL", "HIST", "BIO", "ECON", "PHIL", "STAT"]
        self.course_numbers = ["101", "102", "201", "301", "350", "401", "450", "499"]
        self.titles = {
            "CS": ["Intro to Programming", "Data Structures", "Algorithms", "Computer Systems"],
            "MATH": ["Calculus I", "Calculus II", "Linear Algebra", "Statistics"],
            "PHYS": ["Physics I", "Physics II", "Mechanics", "Electromagnetism"],
            "CHEM": ["General Chemistry", "Organic Chemistry", "Physical Chemistry", "Biochemistry"],
            "ENGL": ["English Composition", "Literature", "Creative Writing", "Technical Writing"],
            "HIST": ["World History", "American History", "European History", "Modern History"],
            "BIO": ["Biology I", "Biology II", "Genetics", "Molecular Biology"],
            "ECON": ["Microeconomics", "Macroeconomics", "Economic Theory", "International Economics"],
            "PHIL": ["Introduction to Philosophy", "Ethics", "Logic", "Philosophy of Mind"],
            "STAT": ["Statistics I", "Statistics II", "Probability", "Statistical Methods"]
        }
        
        self.buildings = ["Science Hall", "Engineering Bldg", "Liberal Arts", "Business Center", "Tech Center"]
        self.room_numbers = ["101", "102", "205", "301", "Lab1", "Aud100", "205A", "B12"]
        
        # Day patterns mapping
        self.day_patterns = {
            "MWF": "MWF", "MW": "MW", "TR": "TR", "TTH": "TR", "MTWF": "MTWF",
            "M": "M", "T": "T", "W": "W", "R": "R", "F": "F"
        }
        
        # Time slots (start, end, duration in minutes)
        self.time_slots = [
            ("0800", "0950", 110), ("0900", "1050", 110), ("1000", "1150", 110),
            ("1100", "1250", 110), ("1300", "1450", 110), ("1400", "1550", 110),
            ("1500", "1650", 110), ("0800", "0850", 50), ("0900", "0950", 50),
            ("1000", "1050", 50), ("1100", "1150", 50), ("1300", "1350", 50),
            ("1400", "1450", 50), ("1500", "1550", 50)
        ]
        
        self.first_names = ["John", "Sarah", "Mike", "Lisa", "David", "Anna", "Chris", "Maria", "Robert", "Jennifer"]
        self.last_names = ["Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas"]
        
        self.credits = ["3", "4", "5"]
        self.limits = ["25", "30", "35", "40", "50", "75", "100"]
        
        # Class types and their typical duration multipliers
        self.class_types = {
            "Lec": {"min_multiplier": 1.0, "suffix": "1"},
            "Lab": {"min_multiplier": 1.5, "suffix": "L1"},
            "Rec": {"min_multiplier": 0.5, "suffix": "R1"},
            "Dis": {"min_multiplier": 0.5, "suffix": "D1"}
        }

    def calculate_min_per_week(self, days: str, duration_minutes: int) -> int:
        """Calculate minutes per week based on days and duration"""
        day_count = len(days)
        return day_count * duration_minutes

    def parse_nlp_input(self, nlp_text: str) -> Dict[str, Any]:
        """Parse NLP input to extract course information"""
        parsed = {}
        
        # Extract subject and course number (e.g., "PHYS 499", "CS 101")
        subject_match = re.search(r'\b([A-Z]{2,4})\s+(\d{3})\b', nlp_text)
        if subject_match:
            parsed['subject'] = subject_match.group(1)
            parsed['course_number'] = subject_match.group(2)
        
        # Extract title (look for common patterns)
        title_patterns = [
            r'titled\s+"([^"]+)"',
            r'titled\s+([A-Za-z\s]+?)(?:\s+with|\s+lecture|\s+lab|\s+MW|\s+TR|\s+MWF|\s+\d+|$)',
            r'(?:course|class)\s+([A-Za-z\s]+?)(?:\s+with|\s+lecture|\s+lab|\s+MW|\s+TR|\s+MWF|\s+\d+|$)'
        ]
        
        for pattern in title_patterns:
            title_match = re.search(pattern, nlp_text, re.IGNORECASE)
            if title_match:
                parsed['title'] = title_match.group(1).strip()
                break
        
        # Extract days (MW, TR, MWF, etc.)
        days_match = re.search(r'\b(MW|TR|MWF|MTWF|TTH|M|T|W|R|F)\b', nlp_text)
        if days_match:
            parsed['days'] = self.day_patterns.get(days_match.group(1), days_match.group(1))
        
        # Extract times (1300 to 1450, 1:00-2:50, etc.)
        time_patterns = [
            r'(\d{4})\s+to\s+(\d{4})',
            r'(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})',
            r'from\s+(\d{4})\s+to\s+(\d{4})'
        ]
        
        for pattern in time_patterns:
            time_match = re.search(pattern, nlp_text)
            if time_match:
                if len(time_match.groups()) == 2:  # 24-hour format
                    parsed['start_time'] = time_match.group(1)
                    parsed['end_time'] = time_match.group(2)
                elif len(time_match.groups()) == 4:  # 12-hour format
                    start_hour = int(time_match.group(1))
                    start_min = time_match.group(2)
                    end_hour = int(time_match.group(3))
                    end_min = time_match.group(4)
                    
                    # Convert to 24-hour format (assume afternoon if > 7)
                    if start_hour < 8:
                        start_hour += 12
                    if end_hour < 8 or end_hour < start_hour:
                        end_hour += 12
                        
                    parsed['start_time'] = f"{start_hour:02d}{start_min}"
                    parsed['end_time'] = f"{end_hour:02d}{end_min}"
                break
        
        # Extract building and room
        building_room_patterns = [
            r'in\s+([A-Za-z\s]+?)\s+room\s+([A-Za-z0-9]+)',
            r'room\s+([A-Za-z0-9]+)\s+in\s+([A-Za-z\s]+)',
            r'in\s+([A-Za-z\s]+?)\s+([A-Za-z0-9]{2,})',
        ]
        
        for pattern in building_room_patterns:
            location_match = re.search(pattern, nlp_text)
            if location_match:
                if 'room' in nlp_text.lower():
                    parsed['building'] = location_match.group(1).strip()
                    parsed['room'] = location_match.group(2).strip()
                else:
                    parsed['building'] = location_match.group(1).strip()
                    parsed['room'] = location_match.group(2).strip()
                break
        
        # Extract instructor
        instructor_patterns = [
            r'(?:Professor|Prof\.?|Dr\.?)\s+([A-Za-z]+)',
            r'with\s+([A-Za-z]+)\s+([A-Za-z]+)',
            r'instructor\s+([A-Za-z]+)\s+([A-Za-z]+)'
        ]
        
        for pattern in instructor_patterns:
            instructor_match = re.search(pattern, nlp_text)
            if instructor_match:
                if len(instructor_match.groups()) == 1:
                    parsed['instructor_lname'] = instructor_match.group(1)
                else:
                    parsed['instructor_fname'] = instructor_match.group(1)
                    parsed['instructor_lname'] = instructor_match.group(2)
                break
        
        # Extract credits
        credit_match = re.search(r'(\d+)\s+credits?', nlp_text)
        if credit_match:
            parsed['credits'] = credit_match.group(1)
        
        # Extract limit
        limit_match = re.search(r'limit\s+(\d+)', nlp_text)
        if limit_match:
            parsed['limit'] = limit_match.group(1)
        
        # Extract class type
        if 'lecture' in nlp_text.lower() or 'lec' in nlp_text.lower():
            parsed['class_type'] = 'Lec'
        elif 'lab' in nlp_text.lower():
            parsed['class_type'] = 'Lab'
        elif 'recitation' in nlp_text.lower() or 'rec' in nlp_text.lower():
            parsed['class_type'] = 'Rec'
        elif 'discussion' in nlp_text.lower() or 'dis' in nlp_text.lower():
            parsed['class_type'] = 'Dis'
        else:
            parsed['class_type'] = 'Lec'  # Default
        
        return parsed

    def fill_missing_data(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in missing data with appropriate defaults"""
        # Subject and course number
        if 'subject' not in parsed:
            parsed['subject'] = random.choice(self.subjects)
        if 'course_number' not in parsed:
            parsed['course_number'] = random.choice(self.course_numbers)
        
        # Title - use subject-appropriate title if not specified
        if 'title' not in parsed:
            subject_titles = self.titles.get(parsed['subject'], self.titles['CS'])
            parsed['title'] = random.choice(subject_titles)
        
        # Time and days
        if 'start_time' not in parsed or 'end_time' not in parsed:
            start_time, end_time, duration = random.choice(self.time_slots)
            parsed['start_time'] = start_time
            parsed['end_time'] = end_time
            parsed['duration'] = duration
        else:
            # Calculate duration from start and end times
            start_hour = int(parsed['start_time'][:2])
            start_min = int(parsed['start_time'][2:])
            end_hour = int(parsed['end_time'][:2])
            end_min = int(parsed['end_time'][2:])
            
            start_total = start_hour * 60 + start_min
            end_total = end_hour * 60 + end_min
            parsed['duration'] = end_total - start_total
        
        if 'days' not in parsed:
            parsed['days'] = random.choice(list(self.day_patterns.keys()))
        
        # Location
        if 'building' not in parsed:
            parsed['building'] = random.choice(self.buildings)
        if 'room' not in parsed:
            parsed['room'] = random.choice(self.room_numbers)
        
        # Instructor
        if 'instructor_fname' not in parsed:
            parsed['instructor_fname'] = random.choice(self.first_names)
        if 'instructor_lname' not in parsed:
            parsed['instructor_lname'] = random.choice(self.last_names)
        
        # Credits and limit
        if 'credits' not in parsed:
            parsed['credits'] = random.choice(self.credits)
        if 'limit' not in parsed:
            parsed['limit'] = random.choice(self.limits)
        
        # Calculate min per week
        parsed['min_per_week'] = self.calculate_min_per_week(parsed['days'], parsed['duration'])
        
        # Generate IDs
        parsed['offering_id'] = str(random.randint(1000, 9999))
        parsed['class_id'] = str(random.randint(100, 999))
        parsed['instructor_id'] = str(random.randint(1000, 9999))
        
        return parsed

    def generate_xml_from_parsed(self, parsed: Dict[str, Any]) -> str:
        """Generate XML from parsed NLP data"""
        class_type = parsed.get('class_type', 'Lec')
        suffix = self.class_types[class_type]['suffix']
        
        xml_template = f'''<?xml version="1.0" ?>
<offerings campus="{self.campus}" year="{self.year}" term="{self.term}" dateFormat="{self.date_format}" created="{self.created}">
  <offering id="{parsed['offering_id']}" offered="true">
    <courseCredit creditType="collegiate" creditUnitType="semesterHours" creditFormat="fixedUnit" fixedCredit="{parsed['credits']}"/>
    <course subject="{parsed['subject']}" courseNbr="{parsed['course_number']}" controlling="true" title="{parsed['title']}"/>
    <config name="1" limit="{parsed['limit']}">
      <subpart type="{class_type}" minPerWeek="{parsed['min_per_week']}"/>
      <class id="{parsed['class_id']}" type="{class_type}" suffix="{suffix}" limit="{parsed['limit']}">
        <time days="{parsed['days']}" startTime="{parsed['start_time']}" endTime="{parsed['end_time']}"/>
        <room building="{parsed['building']}" roomNbr="{parsed['room']}"/>
        <instructor id="{parsed['instructor_id']}" fname="{parsed['instructor_fname']}" lname="{parsed['instructor_lname']}" lead="true"/>
      </class>
    </config>
  </offering>
</offerings>'''
        
        return xml_template

    def process_nlp_to_xml(self, nlp_text: str) -> str:
        """Main method to convert NLP to XML"""
        parsed = self.parse_nlp_input(nlp_text)
        parsed = self.fill_missing_data(parsed)
        xml_output = self.generate_xml_from_parsed(parsed)
        return xml_output

    def generate_training_samples(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Generate training samples with various NLP patterns"""
        samples = []
        
        # Various NLP patterns
        patterns = [
            "Set up {subject} {course_nbr} {title} lecture {days} {start_time} to {end_time} in {building} room {room} with Professor {instructor_lname} {credits} credits limit {limit}",
            "Create course {subject} {course_nbr} titled {title} with {credits} credit hours lecture only limit {limit} students",
            "Schedule {subject} {course_nbr} {title} {days} from {start_time} to {end_time} room {room} in {building} instructor {instructor_fname} {instructor_lname} {credits} credits",
            "Add course {subject} {course_nbr} {title} lecture {days} {start_time}-{end_time} {building} {room} Prof {instructor_lname} limit {limit}",
            "I need to create {subject} {course_nbr} {title} meeting {days} {start_time} to {end_time} in {building} room {room} with {instructor_fname} {instructor_lname} {credits} credits",
            "Can you schedule {subject} {course_nbr} {title} lecture {days} {start_time}-{end_time} {building} room {room} Professor {instructor_lname} limit {limit} students",
            "Please add {subject} {course_nbr} {title} {days} {start_time} to {end_time} {building} {room} {credits} credits limit {limit}",
            "{subject} {course_nbr} {title} lecture {days} {start_time} to {end_time} in {building} room {room} with Professor {instructor_lname} {credits} credits limit {limit}"
        ]
        
        for _ in range(count):
            # Generate random course data
            subject = random.choice(self.subjects)
            course_nbr = random.choice(self.course_numbers)
            title = random.choice(self.titles[subject])
            days = random.choice(list(self.day_patterns.keys()))
            start_time, end_time, duration = random.choice(self.time_slots)
            building = random.choice(self.buildings)
            room = random.choice(self.room_numbers)
            instructor_fname = random.choice(self.first_names)
            instructor_lname = random.choice(self.last_names)
            credits = random.choice(self.credits)
            limit = random.choice(self.limits)
            
            # Choose a random pattern
            pattern = random.choice(patterns)
            
            # Fill in the pattern
            nlp_text = pattern.format(
                subject=subject,
                course_nbr=course_nbr,
                title=title,
                days=days,
                start_time=start_time,
                end_time=end_time,
                building=building,
                room=room,
                instructor_fname=instructor_fname,
                instructor_lname=instructor_lname,
                credits=credits,
                limit=limit
            )
            
            # Generate corresponding XML
            xml_output = self.process_nlp_to_xml(nlp_text)
            
            samples.append({
                "context": "OFFERINGS_REQUEST",
                "input": nlp_text,
                "output": xml_output
            })
        
        return samples

    def split_and_save_dataset(self, samples: List[Dict[str, Any]], output_dir="data/offerings_data"):
        """Split dataset and save to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Split into train/val/test (70/15/15)
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
        
        return splits

    def generate_and_save(self, count: int = 1000):
        """Generate, split, and save the dataset"""
        print(f"Generating {count} training samples...")
        samples = self.generate_training_samples(count)
        
        print("Splitting and saving dataset...")
        splits = self.split_and_save_dataset(samples)
        
        # Show sample
        print("\nSample conversations:")
        print("=" * 80)
        for i, sample in enumerate(samples[:3]):
            print(f"\nExample {i+1}:")
            print(f"Input: {sample['input']}")
            print(f"Output: {sample['output'][:200]}...")
            print("-" * 40)
        
        return samples

# Test the specific case from the original question
def test_specific_case():
    generator = NLPToXMLOfferingsGenerator()
    
    # Test the exact input from the question
    test_input = "Set up PHYS 499 Calculus I lecture MW 1300 to 1450 in Tech Center room Lab1 with Professor Smith 4 credits limit 40"
    
    print("Testing specific case:")
    print(f"Input: {test_input}")
    print("\nGenerated XML:")
    xml_output = generator.process_nlp_to_xml(test_input)
    print(xml_output)

# Usage
def generate_offerings_dataset():
    # Test specific case first
    test_specific_case()
    
    print("\n" + "="*80 + "\n")
    
    # Generate full dataset
    generator = NLPToXMLOfferingsGenerator()
    samples = generator.generate_and_save(1000)  # Generate 1000 samples
# if __name__ == "__main__":
#     generate_offerings_dataset()