# src/data_generation.py
import json
import random
from datetime import datetime, timedelta
import itertools
import os

class UniTimeDatasetGenerator:
    def __init__(self):
        # Sample data for generating realistic examples
        self.subjects = ['CS', 'MATH', 'PHYS', 'CHEM', 'BIOL', 'ENG', 'HIST', 'PSYC', 'ECON', 'ART']
        self.course_numbers = ['101', '102', '201', '202', '301', '302', '401', '402', '499']
        self.instructors = [
            ('Dr.', 'Smith'), ('Prof.', 'Johnson'), ('Dr.', 'Williams'), ('Professor', 'Brown'),
            ('Dr.', 'Jones'), ('Prof.', 'Garcia'), ('Dr.', 'Miller'), ('Professor', 'Davis'),
            ('Dr.', 'Rodriguez'), ('Prof.', 'Martinez'), ('Dr.', 'Anderson'), ('Professor', 'Taylor')
        ]
        self.buildings = ['A', 'B', 'C', 'EDUC', 'SCI', 'ENG', 'LIB', 'THTR', 'GYM']
        self.room_numbers = ['101', '102', '103', '201', '202', '203', '301', '302']
        self.days_combinations = ['MWF', 'TR', 'MW', 'WF', 'MTWRF', 'M', 'T', 'W', 'R', 'F']
        self.time_slots = [
            ('0800', '0900'), ('0900', '1000'), ('1000', '1100'), ('1100', '1200'),
            ('1200', '1300'), ('1300', '1400'), ('1400', '1500'), ('1500', '1600'),
            ('1600', '1700'), ('1700', '1800'), ('1800', '1900'), ('1900', '2000')
        ]
        self.class_types = ['LEC', 'LAB', 'REC', 'SEM', 'IND', 'STU']
        self.preference_levels = ['P', '-2', '-1', '0', '1', '2', 'R']
        self.departments = ['0100', '0101', '0102', '0103', '0104']
        self.majors = ['CS', 'MATH', 'ENG', 'BIOL', 'CHEM', 'PHYS', 'ART', 'HIST']
        self.classifications = ['FR', 'SO', 'JR', 'SR', 'GR']
        self.student_groups = ['HONORS', 'ATHLETES', 'INTERNATIONAL', 'VETERANS', 'TRANSFER']
        
        # Course titles mapping
        self.course_titles = {
            'CS101': 'Introduction to Programming',
            'CS102': 'Data Structures',
            'CS201': 'Algorithms',
            'CS202': 'Database Systems',
            'CS301': 'Software Engineering',
            'CS302': 'Computer Networks',
            'MATH101': 'College Algebra',
            'MATH102': 'Calculus I',
            'MATH201': 'Calculus II',
            'MATH202': 'Linear Algebra',
            'PHYS101': 'General Physics I',
            'PHYS102': 'General Physics II',
            'CHEM101': 'General Chemistry',
            'BIOL101': 'Introduction to Biology',
            'ENG101': 'English Composition',
            'HIST101': 'World History'
        }

    def generate_course_offering_data(self, num_samples=500):
        """Generate CourseOfferingExport.dtd training data"""
        data = []
        
        for _ in range(num_samples):
            subject = random.choice(self.subjects)
            course_num = random.choice(self.course_numbers)
            course_key = f"{subject}{course_num}"
            title = self.course_titles.get(course_key, f"{subject} Course")
            instructor = random.choice(self.instructors)
            days = random.choice(self.days_combinations)
            start_time, end_time = random.choice(self.time_slots)
            building = random.choice(self.buildings)
            room = random.choice(self.room_numbers)
            limit = random.choice([20, 25, 30, 35, 40, 50])
            class_type = random.choice(['LEC', 'SEM'])
            
            # Convert time to readable format for input
            start_readable = self.time_to_readable(start_time)
            end_readable = self.time_to_readable(end_time)
            days_readable = self.days_to_readable(days)
            
            # Natural language input
            input_text = f"Create course offering {subject}{course_num} {title} with {instructor[0]} {instructor[1]} on {days_readable} from {start_readable} to {end_readable} in room {building}{room} with limit {limit} students"
            
            # XML output
            output_xml = f'<offerings campus="MAIN" year="2024" term="Fall"><offering id="{random.randint(1000, 9999)}" offered="true"><course id="{random.randint(100, 999)}" subject="{subject}" courseNbr="{course_num}" title="{title}"><class id="{random.randint(10000, 99999)}" suffix="1" type="{class_type}" limit="{limit}"><time days="{days}" startTime="{start_time}" endTime="{end_time}"/><room building="{building}" roomNbr="{room}"/><instructor id="{random.randint(1, 100)}" fname="{instructor[0]}" lname="{instructor[1]}" lead="true"/></class></course></offering></offerings>'
            
            data.append({
                "input": input_text,
                "output": output_xml,
                "type": "course_offering"
            })
            
        return data

    def generate_course_timetable_data(self, num_samples=500):
        """Generate CourseTimetable.dtd training data"""
        data = []
        
        for _ in range(num_samples):
            subject = random.choice(self.subjects)
            course_num = random.choice(self.course_numbers)
            course_key = f"{subject}{course_num}"
            title = self.course_titles.get(course_key, f"{subject} Course")
            instructor = random.choice(self.instructors)
            days = random.choice(self.days_combinations)
            start_time, end_time = random.choice(self.time_slots)
            building = random.choice(self.buildings)
            room = random.choice(self.room_numbers)
            class_id = random.randint(10000, 99999)
            class_type = random.choice(self.class_types)
            
            # Convert time to readable format for input
            start_readable = self.time_to_readable(start_time)
            end_readable = self.time_to_readable(end_time)
            days_readable = self.days_to_readable(days)
            
            # Handle arranged time classes
            if random.random() < 0.1:  # 10% arranged time classes
                min_per_week = random.choice([1, 2, 3, 4, 5])
                input_text = f"Schedule {subject}{course_num} {title} with {instructor[0]} {instructor[1]} as arranged time with minimum {min_per_week} hours per week, class ID {class_id}"
                output_xml = f'<timetable campus="MAIN" year="2024" term="Fall"><class id="{class_id}" name="{title}" subject="{subject}" courseNbr="{course_num}" type="{class_type}" suffix="1"><arrangeTime minPerWeek="{min_per_week}"/><instructor id="{random.randint(1, 100)}" fname="{instructor[0]}" lname="{instructor[1]}" lead="true"/></class></timetable>'
            else:
                input_text = f"Schedule {subject}{course_num} {title} with {instructor[0]} {instructor[1]} on {days_readable} from {start_readable} to {end_readable} in room {building}{room}, class ID {class_id}"
                output_xml = f'<timetable campus="MAIN" year="2024" term="Fall"><class id="{class_id}" name="{title}" subject="{subject}" courseNbr="{course_num}" type="{class_type}" suffix="1"><time days="{days}" startTime="{start_time}" endTime="{end_time}"/><room building="{building}" roomNbr="{room}"/><instructor id="{random.randint(1, 100)}" fname="{instructor[0]}" lname="{instructor[1]}" lead="true"/></class></timetable>'
            
            data.append({
                "input": input_text,
                "output": output_xml,
                "type": "course_timetable"
            })
            
        return data

    def generate_preferences_data(self, num_samples=500):
        """Generate Preferences.dtd training data"""
        data = []
        
        for _ in range(num_samples):
            preference_type = random.choice(['instructor_time', 'instructor_room', 'department_room', 'distribution'])
            
            if preference_type == 'instructor_time':
                instructor = random.choice(self.instructors)
                dept = random.choice(self.departments)
                level = random.choice(self.preference_levels)
                days = random.choice(['M', 'T', 'W', 'R', 'F', 'MTWRF'])
                start_time, end_time = random.choice(self.time_slots)
                
                level_text = self.level_to_text(level)
                start_readable = self.time_to_readable(start_time)
                end_readable = self.time_to_readable(end_time)
                day_readable = self.day_to_readable(days)
                
                input_text = f"Instructor {instructor[0]} {instructor[1]} from department {dept} {level_text} teaching on {day_readable} from {start_readable} to {end_readable}"
                output_xml = f'<preferences term="Fall" year="2024" campus="MAIN"><instructor externalId="{random.randint(100, 999)}" firstName="{instructor[0]}" lastName="{instructor[1]}" department="{dept}"><timePref level="R"><pref level="{level}" day="{days}" start="{start_time}" stop="{end_time}"/></timePref></instructor></preferences>'
                
            elif preference_type == 'instructor_room':
                instructor = random.choice(self.instructors)
                building = random.choice(self.buildings)
                room = random.choice(self.room_numbers)
                level = random.choice(['P', '-2', '-1', '0', '1', '2'])
                
                level_text = self.level_to_text(level)
                input_text = f"Instructor {instructor[0]} {instructor[1]} {level_text} room {building}{room}"
                output_xml = f'<preferences term="Fall" year="2024" campus="MAIN"><instructor firstName="{instructor[0]}" lastName="{instructor[1]}"><roomPref building="{building}" room="{room}" level="{level}"/></instructor></preferences>'
                
            elif preference_type == 'department_room':
                dept = random.choice(self.departments)
                building = random.choice(self.buildings)
                room = random.choice(self.room_numbers)
                level = random.choice(['P', '-2', '-1', '0', '1', '2'])
                
                level_text = self.level_to_text(level)
                input_text = f"Department {dept} {level_text} room {building}{room}"
                output_xml = f'<preferences term="Fall" year="2024" campus="MAIN"><department code="{dept}"><roomPref building="{building}" room="{room}" level="{level}"/></department></preferences>'
                
            else:  # distribution
                subject = random.choice(self.subjects)
                course_num = random.choice(self.course_numbers)
                dist_type = random.choice(['SAME_ROOM', 'SAME_DAYS', 'PRECEDENCE', 'MAX_HRS_DAY(6)'])
                level = random.choice(['-2', '-1', '1', '2', 'R'])
                
                level_text = self.level_to_text(level)
                input_text = f"{subject}{course_num} lecture and lab should have {dist_type.lower().replace('_', ' ')} constraint with {level_text} level"
                output_xml = f'<preferences term="Fall" year="2024" campus="MAIN"><department code="{random.choice(self.departments)}"><distributionPref type="{dist_type}" structure="Progressive" level="{level}"><subpart subject="{subject}" course="{course_num}" type="Lec"/><subpart subject="{subject}" course="{course_num}" type="Lab"/></distributionPref></department></preferences>'
            
            data.append({
                "input": input_text,
                "output": output_xml,
                "type": "preferences"
            })
            
        return data

    def generate_reservations_data(self, num_samples=500):
        """Generate Reservation.dtd training data"""
        data = []
        
        for _ in range(num_samples):
            subject = random.choice(self.subjects)
            course_num = random.choice(self.course_numbers)
            limit = random.choice([5, 10, 15, 20, 25, 30])
            reservation_type = random.choice(['individual', 'group', 'curriculum'])
            
            if reservation_type == 'individual':
                student_id = f"STU{random.randint(10000, 99999)}"
                expire_date = "2024-12-01"
                
                input_text = f"Reserve {limit} seats in {subject}{course_num} for student with ID {student_id} until {expire_date}"
                output_xml = f'<reservations campus="MAIN" year="2024" term="Fall"><reservation subject="{subject}" courseNbr="{course_num}" limit="{limit}" expire="{expire_date}" type="individual"><student externalId="{student_id}"/></reservation></reservations>'
                
            elif reservation_type == 'group':
                group = random.choice(self.student_groups)
                class_suffix = random.choice(['1', '2', '3'])
                
                input_text = f"Reserve {limit} seats in {subject}{course_num} section {class_suffix} for {group.lower()} student group"
                output_xml = f'<reservations campus="MAIN" year="2024" term="Fall"><reservation subject="{subject}" courseNbr="{course_num}" limit="{limit}" type="group"><class suffix="{class_suffix}"/><studentGroup code="{group}"/></reservation></reservations>'
                
            else:  # curriculum
                major = random.choice(self.majors)
                classification = random.choice(self.classifications)
                
                input_text = f"Reserve {limit} seats in {subject}{course_num} for {major} majors with {classification} classification"
                output_xml = f'<reservations campus="MAIN" year="2024" term="Fall"><reservation subject="{subject}" courseNbr="{course_num}" limit="{limit}" type="curriculum"><academicClassification code="{classification}"/><major code="{major}"/></reservation></reservations>'
            
            data.append({
                "input": input_text,
                "output": output_xml,
                "type": "reservations"
            })
            
        return data

    def time_to_readable(self, time_str):
        """Convert 24-hour time to readable format"""
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        if hour == 0:
            return f"12:{minute:02d} AM"
        elif hour < 12:
            return f"{hour}:{minute:02d} AM"
        elif hour == 12:
            return f"12:{minute:02d} PM"
        else:
            return f"{hour-12}:{minute:02d} PM"

    def days_to_readable(self, days):
        """Convert day codes to readable format"""
        day_map = {'M': 'Monday', 'T': 'Tuesday', 'W': 'Wednesday', 'R': 'Thursday', 'F': 'Friday'}
        if len(days) == 1:
            return day_map[days]
        elif days == 'MWF':
            return 'Monday Wednesday Friday'
        elif days == 'TR':
            return 'Tuesday Thursday'
        elif days == 'MW':
            return 'Monday Wednesday'
        elif days == 'WF':
            return 'Wednesday Friday'
        elif days == 'MTWRF':
            return 'Monday Tuesday Wednesday Thursday Friday'
        else:
            return ' '.join([day_map[d] for d in days])

    def day_to_readable(self, day):
        """Convert single day code to readable format"""
        day_map = {'M': 'Monday', 'T': 'Tuesday', 'W': 'Wednesday', 'R': 'Thursday', 'F': 'Friday', 'MTWRF': 'weekdays'}
        return day_map.get(day, day)

    def level_to_text(self, level):
        """Convert preference level to natural language"""
        level_map = {
            'P': 'prohibits',
            '-2': 'strongly avoids',
            '-1': 'avoids',
            '0': 'is neutral about',
            '1': 'prefers',
            '2': 'strongly prefers',
            'R': 'requires'
        }
        return level_map.get(level, 'prefers')

    def generate_complete_dataset(self, samples_per_type=500):
        """Generate complete dataset with all four DTD types"""
        print("Generating CourseOffering data...")
        course_offering_data = self.generate_course_offering_data(samples_per_type)
        
        print("Generating CourseTimetable data...")
        course_timetable_data = self.generate_course_timetable_data(samples_per_type)
        
        print("Generating Preferences data...")
        preferences_data = self.generate_preferences_data(samples_per_type)
        
        print("Generating Reservations data...")
        reservations_data = self.generate_reservations_data(samples_per_type)
        
        # Combine all data
        complete_data = (course_offering_data + course_timetable_data + 
                        preferences_data + reservations_data)
        
        # Shuffle the combined dataset
        random.shuffle(complete_data)
        
        return complete_data

    def save_dataset(self, data, filename="unitime_complete_dataset.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {filename}")
        print(f"Total samples: {len(data)}")
        
        # Print statistics
        type_counts = {}
        for item in data:
            dtype = item['type']
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        print("\nDataset composition:")
        for dtype, count in type_counts.items():
            print(f"  {dtype}: {count} samples ({count/len(data)*100:.1f}%)")

    # def create_train_test_split(self, data, train_ratio=0.8):
    #     """Split dataset into train and test sets"""
    #     random.shuffle(data)
    #     split_idx = int(len(data) * train_ratio)
        
    #     train_data = data[:split_idx]
    #     test_data = data[split_idx:]
        
    #     return train_data, test_data
    def create_train_val_test_split(self, data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Split dataset into train, validation, and test sets.
        Default: 80% train, 10% validation, 10% test
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        random.shuffle(data)
        total = len(data)
        
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data

# Usage example
if __name__ == "__main__":
    generator = UniTimeDatasetGenerator()
    
    # Generate complete dataset (2000 samples total, 500 per type)
    print("Starting dataset generation...")
    complete_dataset = generator.generate_complete_dataset(samples_per_type=500)
    
    # Split into train/test/validation sets
    print("Creating train/validation/test split...")
    train_data, val_data, test_data = generator.create_train_val_test_split(complete_dataset)


    # Save datasets
    # generator.save_dataset(train_data, "unitime_train_dataset.json")
    # generator.save_dataset(test_data, "unitime_test_dataset.json")
    # generator.save_dataset(complete_dataset, "unitime_complete_dataset.json")
    
    print("\nDataset generation completed!")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    print(f"Validation samples: {len(val_data)}")
    # # Create output directory
    os.makedirs("data/processed", exist_ok=True)

    # # Save datasets to processed folder
    with open("data/processed/train_dataset.json", "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open("data/processed/val_dataset.json", "w") as f:
        json.dump(val_data, f, indent=2,ensure_ascii=False)

    with open("data/processed/test_datas/et.json", "w") as f:
        json.dump(test_data, f, indent=2,ensure_ascii=False)

    with open("data/processed/complete_dataset.json", "w") as f:
        json.dump(complete_dataset, f, indent=2,ensure_ascii=False)

    # # Compute stats
    stats = {
        "total_examples": len(complete_dataset),
        "training_examples": len(train_data),
        "validation_examples": len(val_data),
        "testing_examples": len(test_data),
        "preference_levels_covered": sorted(list(set(entry.get("level", "UNKNOWN") for entry in complete_dataset))),
        "constraint_types": sorted(list(set(entry.get("type", "UNKNOWN") for entry in complete_dataset)))
    }

    # # Save stats
    with open("data/processed/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2,ensure_ascii=False)

    # # Print status
    print("\n✅ Dataset generation completed!")
    print(f"📊 Total examples: {stats['total_examples']}")
    print(f"🎯 Training samples: {stats['training_examples']}")
    print(f"🧪 Validation samples: {stats['validation_examples']}")
    print(f"🧾 Testing samples: {stats['testing_examples']}")
    print(f"🔧 Preference levels: {len(stats['preference_levels_covered'])} → {stats['preference_levels_covered']}")
    print(f"🧩 Constraint types: {len(stats['constraint_types'])} → {stats['constraint_types']}")