import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
import re
import os
import random

class ReservationXMLGenerator:
    def __init__(self):
        self.placeholders = {
            'courseNbr': '[COURSE_NUMBER_PLACEHOLDER]',
            'subject': '[SUBJECT_PLACEHOLDER]',
            'studentId': '[STUDENT_ID_PLACEHOLDER]'
        }

        self.defaults = {
            'campus': 'webegone',
            'term': 'Fal',
            'year': '2010',
            'type': 'individual',
            'dateFormat': 'MM/dd/yyyy'
        }

    def extract_entities_from_nlp(self, nlp_text):
        """Extract entities from natural language text"""
        entities = {}

        # Extract course information - improved pattern to catch subject + number directly
        course_pattern = r'\b([A-Z]{2,4})\s+(\d{3,4})\b'
        course_match = re.search(course_pattern, nlp_text)
        if course_match:
            entities['subject'] = course_match.group(1)
            entities['courseNbr'] = course_match.group(2)
        else:
            # Fallback pattern for "course/class/subject" prefix
            course_pattern_alt = r'(?:course|class|subject)\s+([A-Z]{2,4})\s+(\d{3,4})'
            course_match_alt = re.search(course_pattern_alt, nlp_text, re.IGNORECASE)
            if course_match_alt:
                entities['subject'] = course_match_alt.group(1)
                entities['courseNbr'] = course_match_alt.group(2)

        # Extract student ID
        student_pattern = r'(?:student|id)\s+(\d+)'
        student_match = re.search(student_pattern, nlp_text, re.IGNORECASE)
        if student_match:
            entities['studentId'] = student_match.group(1)

        # Extract limit/capacity
        limit_pattern = r'(?:limit|capacity|max)\s+(\d+)'
        limit_match = re.search(limit_pattern, nlp_text, re.IGNORECASE)
        if limit_match:
            entities['limit'] = limit_match.group(1)

        # Extract term and year
        term_pattern = r'(?:term|semester)\s+(fall|spring|summer|winter)\s*(\d{4})?'
        term_match = re.search(term_pattern, nlp_text, re.IGNORECASE)
        if term_match:
            entities['term'] = term_match.group(1).capitalize()
            if term_match.group(2):
                entities['year'] = term_match.group(2)

        # Extract campus
        campus_pattern = r'(?:campus|location)\s+([A-Za-z\s]+?)(?:\s|$|,)'
        campus_match = re.search(campus_pattern, nlp_text, re.IGNORECASE)
        if campus_match:
            entities['campus'] = campus_match.group(1).strip()

        # Extract reservation type
        if any(word in nlp_text.lower() for word in ['individual', 'personal']):
            entities['type'] = 'individual'
        elif any(word in nlp_text.lower() for word in ['group', 'team']):
            entities['type'] = 'group'
        elif any(word in nlp_text.lower() for word in ['course', 'class']):
            entities['type'] = 'course'
        elif any(word in nlp_text.lower() for word in ['curriculum', 'program']):
            entities['type'] = 'curriculum'

        # Extract major
        major_pattern = r'(?:major|degree)\s+([A-Za-z\s]+?)(?:\s|$|,)'
        major_match = re.search(major_pattern, nlp_text, re.IGNORECASE)
        if major_match:
            entities['major'] = major_match.group(1).strip()

        return entities

    def create_xml_from_nlp(self, nlp_text):
        """Convert NLP text to XML format"""
        entities = self.extract_entities_from_nlp(nlp_text)

        # Create root element
        root = ET.Element("reservations")
        root.set("campus", entities.get('campus', self.defaults['campus']))
        root.set("term", entities.get('term', self.defaults['term']))
        root.set("year", entities.get('year', self.defaults['year']))
        root.set("dateFormat", self.defaults['dateFormat'])
        root.set("created", datetime.now().strftime("%m/%d/%Y"))

        # Create reservation element
        reservation = ET.SubElement(root, "reservation")
        reservation.set("subject", entities.get('subject', self.placeholders['subject']))
        reservation.set("courseNbr", entities.get('courseNbr', self.placeholders['courseNbr']))
        reservation.set("type", entities.get('type', self.defaults['type']))

        # Add optional attributes
        if 'limit' in entities:
            reservation.set("limit", entities['limit'])
        if 'expire' in entities:
            reservation.set("expire", entities['expire'])

        # Add student element if needed
        if 'studentId' in entities:
            student = ET.SubElement(reservation, "student")
            student.set("externalId", entities['studentId'])
        elif entities.get('type', self.defaults['type']) == 'individual':
            student = ET.SubElement(reservation, "student")
            student.set("externalId", self.placeholders['studentId'])

        # Add major element if specified
        if 'major' in entities:
            major = ET.SubElement(reservation, "major")
            major.set("code", entities['major'])

        return self._prettify_xml(root)

    def _prettify_xml(self, elem):
        """Format XML with proper indentation"""
        rough_string = ET.tostring(elem, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    # def generate_training_dataset(self, num_samples=250):
    #     """Generate training dataset for NLP to XML conversion"""
    #     dataset = []

    #     # Sample NLP inputs for training
    #     nlp_examples = [


            
    #         "Reserve [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] in [TERM] [YEAR] at [CAMPUS] campus with limit [LIMIT]",
    #         "Book [SUBJECT] [COURSE_NBR] for Spring 2025",
    #         "Create individual reservation for [SUBJECT] [COURSE_NBR] student ID [STUDENT_ID]",
    #         "Reserve group session for [SUBJECT] [COURSE_NBR] with capacity [LIMIT]",
    #         "Book curriculum reservation for [SUBJECT] [COURSE_NBR] in Summer 2024",
    #         "Reserve [SUBJECT] [COURSE_NBR] for [MAJOR] major student [STUDENT_ID]",
    #         "Individual reservation needed for [SUBJECT] [COURSE_NBR]",
    #         "Group reservation for [SUBJECT] [COURSE_NBR] with limit [LIMIT] students",
    #         "Course reservation for [SUBJECT] [COURSE_NBR] Fall semester",
    #         "Reserve [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] at [CAMPUS] campus",
    #         "Book [SUBJECT] [COURSE_NBR] for [MAJOR] major in Spring 2025",
    #         "Create reservation for [SUBJECT] [COURSE_NBR] with capacity [LIMIT]",
    #         "Reserve [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] at [CAMPUS] campus",
    #         "Individual booking for [SUBJECT] [COURSE_NBR] in Winter 2024",
    #         "Group reservation for [SUBJECT] [COURSE_NBR] limit [LIMIT] students",
    #         "Reserve [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] Fall semester",
    #         "Book [SUBJECT] [COURSE_NBR] curriculum reservation Summer 2025",
    #         "Create [SUBJECT] [COURSE_NBR] reservation for [MAJOR] major",
    #         "Reserve [SUBJECT] [COURSE_NBR] individual session student [STUDENT_ID]",
    #         "Group booking for [SUBJECT] [COURSE_NBR] with limit [LIMIT]",
    #         "Class [SUBJECT] [COURSE_NBR] reserved for student [STUDENT_ID] at [CAMPUS] campus",
    #         "Individual reservation [SUBJECT] [COURSE_NBR] Spring 2025",
    #         "Book [SUBJECT] [COURSE_NBR] for [MAJOR] major student [STUDENT_ID]",
    #         "Create group reservation [SUBJECT] [COURSE_NBR] capacity [LIMIT]",
    #         "Reserve [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] Fall 2024",
    #         "Individual booking [SUBJECT] [COURSE_NBR] at [CAMPUS] campus",
    #         "Reserve [SUBJECT] [COURSE_NBR] curriculum program Summer 2024",
    #         "Group reservation [SUBJECT] [COURSE_NBR] with limit [LIMIT] students",
    #         "Book [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] Spring semester",
    #         "Create reservation [SUBJECT] [COURSE_NBR] for [MAJOR] major"
    #     ]

    #     # Generate dataset entries
    #     for i, nlp_text in enumerate(nlp_examples[:num_samples]):
    #         xml_output = self.create_xml_from_nlp(nlp_text)
    #         dataset.append({
    #             "nlp_input": nlp_text,
    #             "xml_output": xml_output
    #         })

    #     return dataset
    def generate_training_dataset(self, num_samples=250):
        """Generate training dataset for NLP to XML conversion"""
        dataset = []
        
        # Your original 30 examples
        base_examples = [
            "Reserve [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] in [TERM] [YEAR] at [CAMPUS] campus with limit [LIMIT]",
            "Book [SUBJECT] [COURSE_NBR] for Spring 2025",
            "Create individual reservation for [SUBJECT] [COURSE_NBR] student ID [STUDENT_ID]",
            "Reserve group session for [SUBJECT] [COURSE_NBR] with capacity [LIMIT]",
            "Book curriculum reservation for [SUBJECT] [COURSE_NBR] in Summer 2024",
            "Reserve [SUBJECT] [COURSE_NBR] for [MAJOR] major student [STUDENT_ID]",
            "Individual reservation needed for [SUBJECT] [COURSE_NBR]",
            "Group reservation for [SUBJECT] [COURSE_NBR] with limit [LIMIT] students",
            "Course reservation for [SUBJECT] [COURSE_NBR] Fall semester",
            "Reserve [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] at [CAMPUS] campus",
            "Book [SUBJECT] [COURSE_NBR] for [MAJOR] major in Spring 2025",
            "Create reservation for [SUBJECT] [COURSE_NBR] with capacity [LIMIT]",
            "Reserve [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] at [CAMPUS] campus",
            "Individual booking for [SUBJECT] [COURSE_NBR] in Winter 2024",
            "Group reservation for [SUBJECT] [COURSE_NBR] limit [LIMIT] students",
            "Reserve [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] Fall semester",
            "Book [SUBJECT] [COURSE_NBR] curriculum reservation Summer 2025",
            "Create [SUBJECT] [COURSE_NBR] reservation for [MAJOR] major",
            "Reserve [SUBJECT] [COURSE_NBR] individual session student [STUDENT_ID]",
            "Group booking for [SUBJECT] [COURSE_NBR] with limit [LIMIT]",
            "Class [SUBJECT] [COURSE_NBR] reserved for student [STUDENT_ID] at [CAMPUS] campus",
            "Individual reservation [SUBJECT] [COURSE_NBR] Spring 2025",
            "Book [SUBJECT] [COURSE_NBR] for [MAJOR] major student [STUDENT_ID]",
            "Create group reservation [SUBJECT] [COURSE_NBR] capacity [LIMIT]",
            "Reserve [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] Fall 2024",
            "Individual booking [SUBJECT] [COURSE_NBR] at [CAMPUS] campus",
            "Reserve [SUBJECT] [COURSE_NBR] curriculum program Summer 2024",
            "Group reservation [SUBJECT] [COURSE_NBR] with limit [LIMIT] students",
            "Book [SUBJECT] [COURSE_NBR] for student [STUDENT_ID] Spring semester",
            "Create reservation [SUBJECT] [COURSE_NBR] for [MAJOR] major"
        ]
        
        # Simple word replacements to create variations
        replacements = {
            "Reserve": ["Book", "Schedule", "Register", "Secure", "Allocate"],
            "Book": ["Reserve", "Schedule", "Register", "Secure"],
            "Create": ["Make", "Setup", "Establish", "Generate"],
            "reservation": ["booking", "enrollment", "registration", "slot"],
            "individual": ["single", "personal", "private", "one-on-one"],
            "group": ["team", "class", "collective", "multiple"],
            "session": ["class", "course", "meeting", "period"],
            "needed": ["required", "requested", "wanted"],
            "capacity": ["limit", "maximum", "space for", "room for"],
            "limit": ["capacity", "maximum", "max", "up to"],
            "students": ["people", "learners", "participants"],
            "campus": ["location", "site", "branch"],
            "semester": ["term", "session", "period"],
            "curriculum": ["academic", "course", "program"],
            "program": ["curriculum", "course", "academic"],
            "major": ["department", "field", "program"]
        }
        
        # Additional simple templates to add variety
        extra_templates = [
            "Schedule [SUBJECT] [COURSE_NBR] for [STUDENT_ID]",
            "Register [SUBJECT] [COURSE_NBR] in [TERM] [YEAR]",
            "Enroll student [STUDENT_ID] in [SUBJECT] [COURSE_NBR]",
            "Secure spot in [SUBJECT] [COURSE_NBR] for [MAJOR] major",
            "Allocate [SUBJECT] [COURSE_NBR] for [TERM] semester",
            "Setup [SUBJECT] [COURSE_NBR] booking at [CAMPUS]",
            "Request [SUBJECT] [COURSE_NBR] enrollment for [STUDENT_ID]",
            "Need [SUBJECT] [COURSE_NBR] reservation for [MAJOR] student",
            "Want to book [SUBJECT] [COURSE_NBR] for [TERM] [YEAR]",
            "Sign up for [SUBJECT] [COURSE_NBR] at [CAMPUS] campus",
            "[SUBJECT] [COURSE_NBR] enrollment for student [STUDENT_ID]",
            "Course [SUBJECT] [COURSE_NBR] booking for [MAJOR] major",
            "Class registration [SUBJECT] [COURSE_NBR] in [TERM]",
            "Student [STUDENT_ID] needs [SUBJECT] [COURSE_NBR]",
            "[MAJOR] major requesting [SUBJECT] [COURSE_NBR]",
            "Please reserve [SUBJECT] [COURSE_NBR] for [STUDENT_ID]",
            "Can you book [SUBJECT] [COURSE_NBR] for [TERM]?",
            "Looking for [SUBJECT] [COURSE_NBR] spot in [TERM] [YEAR]",
            "Add [STUDENT_ID] to [SUBJECT] [COURSE_NBR]",
            "Enroll in [SUBJECT] [COURSE_NBR] at [CAMPUS] campus"
        ]
        
        def create_variation(text):
            """Create a variation by replacing words"""
            for original, alternatives in replacements.items():
                if original in text:
                    # Only replace sometimes to create variety
                    import random
                    if random.random() < 0.3:  # 30% chance to replace
                        replacement = random.choice(alternatives)
                        text = text.replace(original, replacement, 1)  # Replace only first occurrence
            return text
        
        # Start with original examples
        all_examples = base_examples.copy()
        
        # Add extra templates
        all_examples.extend(extra_templates)
        
        # Create variations of existing examples
        import random
        while len(all_examples) < num_samples:
            # Pick a random base example
            base = random.choice(base_examples + extra_templates)
            
            # Create a variation
            variation = create_variation(base)
            
            # Add it if it's different from the original
            if variation != base and variation not in all_examples:
                all_examples.append(variation)
            
            # If we're stuck, add some manual variations
            if len(all_examples) < num_samples and len(all_examples) % 10 == 0:
                # Add some simple manual variations
                base_example = random.choice(base_examples)
                if "reservation" in base_example:
                    manual_var = base_example.replace("reservation", "booking")
                    if manual_var not in all_examples:
                        all_examples.append(manual_var)
        
        # Generate dataset entries
        for i, nlp_text in enumerate(all_examples[:num_samples]):
            xml_output = self.create_xml_from_nlp(nlp_text)
            dataset.append({
                "input": nlp_text,
                "output": xml_output
            })
        
        return dataset

def generate_reservations_dataset():
    """Main function to generate and save the dataset"""
    generator = ReservationXMLGenerator()

    print("\n=== GENERATING NLP TO XML TRAINING DATASET ===")
    dataset = generator.generate_training_dataset(250)

    # Create output directory
    os.makedirs("data/reservation_data", exist_ok=True)

    # Shuffle dataset
    random.shuffle(dataset)

    # Split dataset into train/validation/test
    total = len(dataset)
    train_split = int(0.8 * total)
    val_split = int(0.10 * total)

    train_data = dataset[:train_split]
    val_data = dataset[train_split:train_split + val_split]
    test_data = dataset[train_split + val_split:]

    # Save datasets
    with open("data/reservation_data/train_reservation.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open("data/reservation_data/val_reservation.json", "w") as f:
        json.dump(val_data, f, indent=2)

    with open("data/reservation_data/test_reservation.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print("Dataset saved to 'data/reservation_data/' directory.")
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    # Print a sample for verification
    print("\n=== SAMPLE TRAINING EXAMPLE ===")
    sample = train_data[0]
    print("NLP Input:", sample["input"])
    print("XML Output:")
    print(sample["output"])

# if __name__ == "__main__":
#     generate_reservations_dataset()
#     print("Reservation dataset generation complete.")