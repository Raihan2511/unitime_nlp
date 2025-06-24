import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
import re
import os
import random

class ReservationXMLGenerator:
    def __init__(self):
        # Sample data pools for realistic values
        self.sample_data = {
            'subjects': [
                "CS", "MATH", "PHYS", "CHEM", "BIOL", "ENG", 
                "HIST", "PSYC", "ECON", "PHIL", "STAT", "ART"
            ],
            'course_numbers': [
                "101", "102", "150", "201", "202", "250", "301", 
                "302", "350", "401", "402", "450", "499"
            ],
            'student_ids': [
                f"{random.randint(100000, 999999)}" for _ in range(100)
            ],
            'campuses': [
                'main', 'north', 'south', 'downtown', 'webegone'
            ],
            'terms': [
                'Fall', 'Spring', 'Summer', 'Winter'
            ],
            'years': [
                '2024', '2025', '2026'
            ],
            'majors': [
                "Computer Science", "Mathematics", "Engineering", 
                "Biology", "Physics", "Chemistry", "Economics", 
                "History", "Psychology", "Philosophy", "Statistics", "Art"
            ],
            'academic_areas': [
                'STEM', 'Liberal Arts', 'Business', 'Engineering'
            ],
            'student_groups': [
                'Honors', 'Athletes', 'International', 'Graduate'
            ],
            'limits': [
                "12", "15", "18", "20", "24","50","30", "36", "40", "45", "60"
            ]
        }


        self.defaults = {
            'campus': 'webegone',
            'term': 'Fall',
            'year': '2024',
            'type': 'individual',
            'dateFormat': 'MM/dd/yyyy'
        }

    def extract_entities_from_nlp(self, nlp_text):
        """Extract entities from natural language text with improved patterns"""
        entities = {}

        # Course information - multiple patterns
        patterns = [
            r'\b([A-Z]{2,4})\s+(\d{3,4})\b',  # CS 101
            r'(?:course|class|subject)\s+([A-Z]{2,4})\s+(\d{3,4})',
            r'([A-Z]{2,4})(\d{3,4})\b'  # CS101
        ]
        
        for pattern in patterns:
            match = re.search(pattern, nlp_text, re.IGNORECASE)
            if match:
                entities['subject'] = match.group(1).upper()
                entities['courseNbr'] = match.group(2)
                break

        # Student ID patterns
        student_patterns = [
            r'(?:student|id)\s+(\d{6,9})',
            r'student\s+id\s+(\d{6,9})',
            r'id\s*:\s*(\d{6,9})'
        ]
        
        for pattern in student_patterns:
            match = re.search(pattern, nlp_text, re.IGNORECASE)
            if match:
                entities['studentId'] = match.group(1)
                break

        # Limit/capacity
        limit_match = re.search(r'(?:limit|capacity|max|up to)\s+(\d+)', nlp_text, re.IGNORECASE)
        if limit_match:
            entities['limit'] = limit_match.group(1)

        # Term and year
        term_match = re.search(r'(?:term|semester)\s+(fall|spring|summer|winter)\s*(\d{4})?', nlp_text, re.IGNORECASE)
        if term_match:
            entities['term'] = term_match.group(1).capitalize()
            if term_match.group(2):
                entities['year'] = term_match.group(2)

        # Standalone year
        if 'year' not in entities:
            year_match = re.search(r'\b(20\d{2})\b', nlp_text)
            if year_match:
                entities['year'] = year_match.group(1)

        # Campus
        campus_match = re.search(r'(?:campus|location)\s+([a-zA-Z]+)', nlp_text, re.IGNORECASE)
        if campus_match:
            entities['campus'] = campus_match.group(1).lower()

        # Reservation type
        if re.search(r'\b(?:individual|personal|single)\b', nlp_text, re.IGNORECASE):
            entities['type'] = 'individual'
        elif re.search(r'\b(?:group|team|multiple)\b', nlp_text, re.IGNORECASE):
            entities['type'] = 'group'
        elif re.search(r'\b(?:course|class)\b', nlp_text, re.IGNORECASE):
            entities['type'] = 'course'
        elif re.search(r'\b(?:curriculum|program)\b', nlp_text, re.IGNORECASE):
            entities['type'] = 'curriculum'

        # Major
        major_match = re.search(r'(?:major|degree)\s+([A-Za-z\s]+?)(?:\s|$|,)', nlp_text, re.IGNORECASE)
        if major_match:
            entities['major'] = major_match.group(1).strip()

        return entities

    def fill_placeholders(self, entities):
        """Fill missing entities with realistic sample data"""
        filled = entities.copy()
        
        # Fill required fields if missing
        if 'subject' not in filled:
            filled['subject'] = random.choice(self.sample_data['subjects'])
        if 'courseNbr' not in filled:
            filled['courseNbr'] = random.choice(self.sample_data['course_numbers'])
        if 'studentId' not in filled and filled.get('type', 'individual') == 'individual':
            filled['studentId'] = random.choice(self.sample_data['student_ids'])
            
        # Fill optional fields randomly (30% chance)
        if random.random() < 0.3 and 'limit' not in filled:
            filled['limit'] = random.choice(self.sample_data['limits'])
        if random.random() < 0.2 and 'major' not in filled:
            filled['major'] = random.choice(self.sample_data['majors'])
        if random.random() < 0.15:
            filled['academicArea'] = random.choice(self.sample_data['academic_areas'])
        if random.random() < 0.1:
            filled['studentGroup'] = random.choice(self.sample_data['student_groups'])
            
        return filled

    def create_xml_from_nlp(self, nlp_text):
        """Convert NLP text to XML format with realistic data"""
        entities = self.extract_entities_from_nlp(nlp_text)
        entities = self.fill_placeholders(entities)

        # Create root element
        root = ET.Element("reservations")
        root.set("campus", entities.get('campus', self.defaults['campus']))
        root.set("term", entities.get('term', self.defaults['term']))
        root.set("year", entities.get('year', self.defaults['year']))
        root.set("dateFormat", self.defaults['dateFormat'])
        root.set("created", datetime.now().strftime("%m/%d/%Y"))

        # Create reservation element
        reservation = ET.SubElement(root, "reservation")
        reservation.set("subject", entities['subject'])
        reservation.set("courseNbr", entities['courseNbr'])
        reservation.set("type", entities.get('type', self.defaults['type']))

        # Add optional attributes
        if 'limit' in entities:
            reservation.set("limit", entities['limit'])

        # Add student element for individual reservations
        if entities.get('type', 'individual') == 'individual' and 'studentId' in entities:
            student = ET.SubElement(reservation, "student")
            student.set("externalId", entities['studentId'])

        # Add optional elements
        if 'major' in entities:
            major = ET.SubElement(reservation, "major")
            major.set("code", entities['major'])

        if 'academicArea' in entities:
            area = ET.SubElement(reservation, "academicArea")
            area.set("abbreviation", entities['academicArea'])

        if 'studentGroup' in entities:
            group = ET.SubElement(reservation, "studentGroup")
            group.set("code", entities['studentGroup'])

        return self._prettify_xml(root)

    def _prettify_xml(self, elem):
        """Format XML with proper indentation"""
        rough_string = ET.tostring(elem, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def generate_diverse_templates(self):
        """Generate diverse conversational templates"""
        templates = [
            # Basic requests
            "I need to reserve {subject} {course_nbr}",
            "Can you book {subject} {course_nbr} for me?",
            "Please reserve {subject} {course_nbr} for student {student_id}",
            "Book {subject} {course_nbr} for {term} {year}",
            
            # Conversational style
            "Hi, I'd like to register for {subject} {course_nbr}",
            "Hello, can I get a spot in {subject} {course_nbr}?",
            "I want to enroll in {subject} {course_nbr} this {term}",
            "Could you help me reserve {subject} {course_nbr}?",
            
            # Specific requirements
            "Reserve {subject} {course_nbr} for {major} major",
            "I need {subject} {course_nbr} with limit {limit}",
            "Book group session for {subject} {course_nbr}",
            "Individual reservation for {subject} {course_nbr}",
            
            # Campus specific
            "Reserve {subject} {course_nbr} at {campus} campus",
            "I need {subject} {course_nbr} at the {campus} location",
            
            # Informal requests
            "Sign me up for {subject} {course_nbr}",
            "Add me to {subject} {course_nbr}",
            "Put me down for {subject} {course_nbr}",
            "I want to take {subject} {course_nbr}",
            
            # Question format
            "Is {subject} {course_nbr} available?",
            "Can I get into {subject} {course_nbr}?",
            "Any spots left in {subject} {course_nbr}?",
            
            # Detailed requests
            "Reserve {subject} {course_nbr} for student ID {student_id} in {term} {year}",
            "Book {subject} {course_nbr} for {major} student {student_id}",
            "Create group reservation for {subject} {course_nbr} with capacity {limit}"
        ]
        return templates

    def generate_training_dataset(self, num_samples=300):
        """Generate diverse training dataset with realistic data"""
        dataset = []
        templates = self.generate_diverse_templates()
        
        for i in range(num_samples):
            template = random.choice(templates)
            
            # Fill template with sample data
            filled_text = template.format(
                subject=random.choice(self.sample_data['subjects']),
                course_nbr=random.choice(self.sample_data['course_numbers']),
                student_id=random.choice(self.sample_data['student_ids']),
                term=random.choice(self.sample_data['terms']),
                year=random.choice(self.sample_data['years']),
                campus=random.choice(self.sample_data['campuses']),
                major=random.choice(self.sample_data['majors']),
                limit=random.choice(self.sample_data['limits'])
            )
            
            # Generate XML output
            xml_output = self.create_xml_from_nlp(filled_text)
            
            dataset.append({
                "input": filled_text,
                "output": xml_output
            })
        
        return dataset

def generate_reservations_dataset():
    """Main function to generate and save the dataset"""
    generator = ReservationXMLGenerator()
    
    print("Generating NLP to XML training dataset...")
    dataset = generator.generate_training_dataset(1000)  # Adjust number of samples as needed
    
    # Create output directory
    os.makedirs("data/reservation_data", exist_ok=True)
    
    # Shuffle and split dataset
    random.shuffle(dataset)
    train_split = int(0.8 * len(dataset))
    val_split = int(0.1 * len(dataset))
    
    train_data = dataset[:train_split]
    val_data = dataset[train_split:train_split + val_split]
    test_data = dataset[train_split + val_split:]
    
    # Save datasets
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        with open(f"data/reservation_data/{name}_reservation.json", "w") as f:
            json.dump(data, f, indent=2)
    
    print(f"Dataset generated: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Show sample
    print("\nSample training example:")
    print("Input:", train_data[0]["input"])
    print("Output:", train_data[0]["output"][:200] + "...")

# if __name__ == "__main__":
#     generate_reservations_dataset()