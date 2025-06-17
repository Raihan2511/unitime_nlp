import json
import random
from datetime import datetime, timedelta
import itertools
import os

class EnhancedUniTimeDatasetGenerator:
    def __init__(self):
        # Enhanced sample data based on your XML examples
        self.subjects = ['CS', 'MATH', 'PHYS', 'CHEM', 'BIOL', 'ENG', 'HIST', 'PSYC', 'ECON', 'ART', 'CHM', 'ALG', 'COM', 'ENGL', 'BAND']
        self.course_numbers = ['101', '102', '201', '202', '301', '302', '401', '402', '499', '101H']
        self.instructors = [
            ('Dr.', 'Smith', 'John'), ('Prof.', 'Johnson', 'Mary'), ('Dr.', 'Williams', 'James'), 
            ('Professor', 'Brown', 'Lisa'), ('Dr.', 'Jones', 'Michael'), ('Prof.', 'Garcia', 'Ana'),
            ('Dr.', 'Miller', 'David'), ('Professor', 'Davis', 'Sarah'), ('Dr.', 'Rodriguez', 'Carlos'),
            ('Prof.', 'Martinez', 'Elena'), ('Dr.', 'Anderson', 'Robert'), ('Professor', 'Taylor', 'Jessica'),
            ('Dr.', 'DOE', 'JOE')
        ]
        self.buildings = ['EDUC', 'THIR', 'SCI', 'ENG', 'LIB', 'THTR', 'GYM', 'MALL']
        self.room_numbers = ['101', '102', '103', '104', '105', '106', '107', '108', '201', '202', '203']
        self.days_combinations = ['MWF', 'TR', 'MW', 'WF', 'MTWRF', 'M', 'T', 'W', 'R', 'F', 'TTh', 'MF']
        self.time_slots = [
            ('0730', '0830'), ('0800', '0900'), ('0830', '0930'), ('0900', '1000'), ('0930', '1020'),
            ('1000', '1100'), ('1030', '1130'), ('1100', '1200'), ('1130', '1230'), ('1200', '1300'),
            ('1230', '1330'), ('1300', '1400'), ('1330', '1430'), ('1400', '1500'), ('1430', '1530'),
            ('1500', '1600'), ('1530', '1630'), ('1600', '1700'), ('1700', '1800'), ('1800', '1900'),
            ('1900', '2000')
        ]
        self.class_types = ['Lec', 'Lab', 'Rec', 'Sem', 'Ind', 'Stu']
        self.preference_levels = ['P', '-2', '-1', '0', '1', '2', 'R']
        self.departments = ['0100', '0101', '0102', '0103', '0104']
        self.majors = ['CS', 'MATH', 'ENG', 'BIOL', 'CHEM', 'PHYS', 'ART', 'HIST', 'M1', 'M2']
        self.classifications = ['01', '02', '03', '04', 'FR', 'SO', 'JR', 'SR', 'GR']
        self.student_groups = ['HONORS', 'ATHLETES', 'INTERNATIONAL', 'VETERANS', 'TRANSFER', 'YEX']
        self.academic_areas = ['A', 'B', 'C', 'ENG', 'SCI']
        self.distribution_types = ['SAME_ROOM', 'SAME_DAYS', 'PRECEDENCE', 'MAX_HRS_DAY(6)', 'SAME_INSTR']
        self.time_patterns = ['1x50', '2x75', '3x50', 'Full Term', 'Even Wks', 'Odd Wks']
        self.date_patterns = ['Full Term', 'Even Wks', 'Odd Wks', 'First Half', 'Second Half']
        
        # Course titles mapping
        self.course_titles = {
            'CS101': 'Introduction to Programming',
            'CS102': 'Data Structures',
            'CS201': 'Algorithms',
            'CS202': 'Database Systems',
            'MATH101': 'College Algebra',
            'MATH102': 'Calculus I',
            'PHYS101': 'General Physics I',
            'CHEM101': 'General Chemistry',
            'CHM101': 'General Chemistry',
            'BIOL101': 'Introduction to Biology',
            'ENG101': 'English Composition',
            'ENGL101': 'English Composition',
            'ALG101': 'College Algebra',
            'COM101': 'Public Speaking',
            'BAND101': 'Concert Band'
        }

    def generate_course_offering_data(self, num_samples=500):
        """Generate CourseOfferingExport.dtd training data"""
        data = []
        
        for _ in range(num_samples):
            subject = random.choice(self.subjects)
            course_num = random.choice(self.course_numbers)
            course_key = f"{subject}{course_num}"
            title = self.course_titles.get(course_key, f"{subject} Course {course_num}")
            
            # Multiple courses in offering
            courses = []
            main_course = {
                'subject': subject,
                'courseNbr': course_num,
                'title': title,
                'controlling': True
            }
            courses.append(main_course)
            
            # Sometimes add cross-listed course
            if random.random() < 0.3:
                alt_course = {
                    'subject': subject,
                    'courseNbr': course_num + 'H',
                    'title': title + ' Honors',
                    'controlling': False
                }
                courses.append(alt_course)
            
            # Generate configurations
            configs = []
            for config_num in range(1, random.randint(2, 4)):
                config = self.generate_config(config_num)
                configs.append(config)
            
            # Natural language input
            course_desc = f"{subject} {course_num} {title}"
            if len(courses) > 1:
                course_desc += f" with cross-listed {courses[1]['subject']} {courses[1]['courseNbr']}"
            
            config_desc = f"with {len(configs)} configuration{'s' if len(configs) > 1 else ''}"
            for i, config in enumerate(configs):
                config_desc += f", config {i+1} has limit {config['limit']}"
            
            input_text = f"Create course offering for {course_desc} {config_desc}"
            
            # XML output
            offering_id = random.randint(10000, 99999)
            output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE offerings SYSTEM "CourseOfferingExport.dtd">\n<offerings campus="woebegon" year="2010" term="Fal">\n    <offering id="{offering_id}" offered="true">\n        <courseCredit creditType="collegiate" creditUnitType="semesterHours" creditFormat="fixedUnit" fixedCredit="3"/>\n'
            
            for course in courses:
                consent = random.choice(['', ' consent="OP"', ' consent="IN"']) if random.random() < 0.2 else ''
                output_xml += f'        <course subject="{course["subject"]}" courseNbr="{course["courseNbr"]}" controlling="{str(course["controlling"]).lower()}" title="{course["title"]}"{consent}/>\n'
            
            for config in configs:
                output_xml += self.generate_config_xml(config)
            
            output_xml += '    </offering>\n</offerings>'
            
            data.append({
                "input": input_text,
                "output": output_xml,
                "type": "course_offering"
            })
            
        return data

    def generate_config(self, config_num):
        """Generate a configuration with subparts and classes"""
        limit = random.choice([15, 20, 25, 30, 35, 40])
        subparts = []
        
        # Main subpart (usually Lecture)
        main_subpart = {
            'type': 'Lec',
            'minPerWeek': random.choice([50, 75, 100, 150]),
            'classes': []
        }
        
        # Add classes to main subpart
        num_classes = random.randint(1, 3)
        for i in range(1, num_classes + 1):
            class_info = {
                'type': 'Lec',
                'suffix': str(i),
                'limit': limit // num_classes,
                'scheduleNote': '',
                'studentScheduling': True
            }
            main_subpart['classes'].append(class_info)
        
        subparts.append(main_subpart)
        
        # Sometimes add Lab or Recitation
        if random.random() < 0.6:
            additional_type = random.choice(['Lab', 'Rec'])
            additional_subpart = {
                'type': additional_type,
                'minPerWeek': random.choice([100, 150, 200]),
                'classes': []
            }
            
            num_additional = random.randint(1, 2)
            for i in range(1, num_additional + 1):
                class_info = {
                    'type': additional_type,
                    'suffix': str(i),
                    'limit': random.choice([10, 15, 20])
                }
                additional_subpart['classes'].append(class_info)
            
            subparts.append(additional_subpart)
        
        return {
            'name': str(config_num),
            'limit': limit,
            'subparts': subparts
        }

    def generate_config_xml(self, config):
        """Generate XML for a configuration"""
        xml = f'        <config name="{config["name"]}" limit="{config["limit"]}">\n'
        
        for subpart in config['subparts']:
            xml += f'            <subpart type="{subpart["type"]}" suffix="" minPerWeek="{subpart["minPerWeek"]}">\n'
            xml += f'                <subpartCredit creditType="collegiate" creditUnitType="semesterHours" creditFormat="fixedUnit" fixedCredit="1"/>\n'
            
            for class_info in subpart['classes']:
                xml += f'                <class id="" type="{class_info["type"]}" suffix="{class_info["suffix"]}" limit="{class_info["limit"]}"'
                if class_info.get('scheduleNote'):
                    xml += f' scheduleNote="{class_info["scheduleNote"]}"'
                if class_info.get('studentScheduling'):
                    xml += f' studentScheduling="{str(class_info["studentScheduling"]).lower()}"'
                xml += '/>\n'
            
            xml += f'            </subpart>\n'
        
        xml += f'        </config>\n'
        return xml

    def generate_course_timetable_data(self, num_samples=500):
        """Generate CourseTimetable.dtd training data"""
        data = []
        
        for _ in range(num_samples):
            subject = random.choice(self.subjects)
            course_num = random.choice(self.course_numbers)
            course_key = f"{subject}{course_num}"
            title = self.course_titles.get(course_key, f"{subject} Course")
            class_type = random.choice(self.class_types)
            suffix = str(random.randint(1, 5))
            
            # Handle arranged time classes
            if random.random() < 0.15:  # 15% arranged time classes
                min_per_week = random.choice([50, 100, 150, 200, 250])
                input_text = f"Schedule {subject} {course_num} {title} {class_type} section {suffix} as arranged time with minimum {min_per_week} minutes per week"
                
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE timetable SYSTEM "CourseTimetable.dtd">\n<timetable campus="woebegon" year="2010" term="Fal" action="update" timeFormat="HHmm" dateFormat="yyyy/M/d">\n    <class subject="{subject}" courseNbr="{course_num}" type="{class_type}" suffix="{suffix}">\n        <arrangeTime minPerWeek="{min_per_week}"/>\n    </class>\n</timetable>'
            else:
                days = random.choice(self.days_combinations)
                start_time, end_time = random.choice(self.time_slots)
                building = random.choice(self.buildings)
                room = random.choice(self.room_numbers)
                
                # Optional time pattern and date pattern
                time_pattern = random.choice(self.time_patterns) if random.random() < 0.3 else None
                date_pattern = random.choice(self.date_patterns) if random.random() < 0.2 else None
                
                # Convert to readable format
                days_readable = self.days_to_readable(days)
                start_readable = self.time_to_readable(start_time)
                end_readable = self.time_to_readable(end_time)
                
                input_text = f"Schedule {subject} {course_num} {title} {class_type} section {suffix} on {days_readable} from {start_readable} to {end_readable} in room {building} {room}"
                
                if time_pattern:
                    input_text += f" with {time_pattern} time pattern"
                if date_pattern:
                    input_text += f" following {date_pattern} schedule"
                
                # Generate XML
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE timetable SYSTEM "CourseTimetable.dtd">\n<timetable campus="woebegon" year="2010" term="Fal" action="update" timeFormat="HHmm" dateFormat="yyyy/M/d">\n    <class subject="{subject}" courseNbr="{course_num}" type="{class_type}" suffix="{suffix}">\n'
                
                # Time element
                time_elem = f'        <time days="{days}" startTime="{start_time}"'
                if end_time:
                    time_elem += f' endTime="{end_time}"'
                if time_pattern:
                    time_elem += f' timePattern="{time_pattern}"'
                if date_pattern:
                    time_elem += f' datePattern="{date_pattern}"'
                time_elem += '/>\n'
                output_xml += time_elem
                
                # Room element
                if random.random() < 0.8:  # 80% have room assignment
                    room_elem = f'        <room building="{building}" roomNbr="{room}"/>\n'
                    output_xml += room_elem
                elif random.random() < 0.5:  # Some use location instead
                    location_elem = f'        <location name="{building}"/>\n'
                    output_xml += location_elem
                
                output_xml += '    </class>\n</timetable>'
            
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
            preference_type = random.choice(['instructor_time', 'instructor_room', 'instructor_feature', 
                                          'department_room', 'department_location', 'distribution'])
            
            if preference_type == 'instructor_time':
                instructor = random.choice(self.instructors)
                dept = random.choice(self.departments)
                ext_id = random.randint(100, 999)
                
                # Generate time preferences for multiple days
                time_prefs = []
                for day in random.sample(['M', 'T', 'W', 'R', 'F'], random.randint(2, 5)):
                    level = random.choice(self.preference_levels)
                    start_time, end_time = random.choice(self.time_slots)
                    time_prefs.append({
                        'day': day,
                        'level': level,
                        'start': start_time,
                        'stop': end_time
                    })
                
                # Natural language description
                level_text = self.level_to_text(time_prefs[0]['level'])
                days_text = ', '.join([self.day_to_readable(tp['day']) for tp in time_prefs[:2]])
                input_text = f"Instructor {instructor[2]} {instructor[1]} from department {dept} {level_text} teaching on {days_text} during morning hours"
                
                # XML output
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE preferences SYSTEM "Preferences.dtd">\n<preferences term="Fal" year="2010" campus="woebegon" dateFormat="yyyy/M/d" timeFormat="HHmm">\n'
                output_xml += f'    <instructor externalId="{ext_id}" firstName="{instructor[2]}" lastName="{instructor[1]}" department="{dept}">\n'
                output_xml += f'        <timePref level="R">\n'
                
                for tp in time_prefs:
                    output_xml += f'            <pref level="{tp["level"]}" day="{tp["day"]}" start="{tp["start"]}" stop="{tp["stop"]}"/>\n'
                
                output_xml += f'        </timePref>\n    </instructor>\n</preferences>'
                
            elif preference_type == 'instructor_feature':
                instructor = random.choice(self.instructors)
                feature = random.choice(['Comp', 'Proj', 'Board', 'Video', 'Audio'])
                level = random.choice(['-2', '-1', '0', '1', '2'])
                
                level_text = self.level_to_text(level)
                input_text = f"Instructor {instructor[2]} {instructor[1]} {level_text} rooms with {feature.lower()} equipment"
                
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE preferences SYSTEM "Preferences.dtd">\n<preferences term="Fal" year="2010" campus="woebegon">\n'
                output_xml += f'    <instructor firstName="{instructor[2]}" lastName="{instructor[1]}">\n'
                output_xml += f'        <featurePref feature="{feature}" level="{level}"/>\n'
                output_xml += f'    </instructor>\n</preferences>'
                
            elif preference_type == 'department_room':
                dept = random.choice(self.departments)
                building = random.choice(self.buildings)
                room = random.choice(self.room_numbers)
                level = random.choice(['P', '-2', '-1', '0', '1', '2'])
                
                level_text = self.level_to_text(level)
                input_text = f"Department {dept} {level_text} using room {building} {room}"
                
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE preferences SYSTEM "Preferences.dtd">\n<preferences term="Fal" year="2010" campus="woebegon">\n'
                output_xml += f'    <department code="{dept}">\n'
                output_xml += f'        <roomPref building="{building}" room="{room}" level="{level}"/>\n'
                output_xml += f'    </department>\n</preferences>'
                
            elif preference_type == 'department_location':
                dept = random.choice(self.departments)
                location = random.choice(self.buildings)
                level = random.choice(['0', '1', '2'])
                
                level_text = self.level_to_text(level)
                input_text = f"Department {dept} {level_text} classes in {location} building area"
                
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE preferences SYSTEM "Preferences.dtd">\n<preferences term="Fal" year="2010" campus="woebegon">\n'
                output_xml += f'    <department code="{dept}">\n'
                output_xml += f'        <roomPref location="{location}" level="{level}"/>\n'
                output_xml += f'    </department>\n</preferences>'
                
            else:  # distribution
                subject = random.choice(self.subjects)
                course_num = random.choice(self.course_numbers)
                dist_type = random.choice(self.distribution_types)
                level = random.choice(['-2', '-1', '1', '2', 'R'])
                structure = random.choice(['Progressive', 'AllClasses', 'Pairwise'])
                
                level_text = self.level_to_text(level)
                constraint_desc = dist_type.lower().replace('_', ' ').replace('(6)', ' max 6 hours')
                
                if structure == 'Progressive':
                    input_text = f"{subject} {course_num} lecture and lab sections should have {constraint_desc} constraint with {level_text} preference"
                    subparts_xml = f'        <subpart subject="{subject}" course="{course_num}" type="Lec"/>\n        <subpart subject="{subject}" course="{course_num}" type="Lab"/>\n'
                else:
                    input_text = f"{subject} {course_num} all recitation sections should have {constraint_desc} constraint with {level_text} preference"
                    subparts_xml = ''.join([f'        <class subject="{subject}" course="{course_num}" type="Rec" suffix="{i}"/>\n' for i in range(1, 4)])
                
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE preferences SYSTEM "Preferences.dtd">\n<preferences term="Fal" year="2010" campus="woebegon">\n'
                output_xml += f'    <distributionPref type="{dist_type}" structure="{structure}" level="{level}">\n'
                output_xml += subparts_xml
                output_xml += f'    </distributionPref>\n</preferences>'
            
            data.append({
                "input": input_text,
                "output": output_xml,
                "type": "preferences"
            })
            
        return data

    def generate_reservations_data(self, num_samples=500):
        """Generate Reservations.dtd training data"""
        data = []
        
        for _ in range(num_samples):
            subject = random.choice(self.subjects)
            course_num = random.choice(self.course_numbers)
            limit = random.choice([2, 3, 5, 7, 10, 12, 15, 20])
            reservation_type = random.choice(['individual', 'group', 'curriculum', 'course'])
            
            if reservation_type == 'individual':
                # Multiple students for individual reservation
                num_students = random.randint(1, 3)
                student_ids = [f"{random.randint(1001, 9999)}" for _ in range(num_students)]
                expire_date = random.choice(["09/01/2010", "09/10/2010", "08/25/2010"])
                class_suffix = str(random.randint(1, 5))
                
                student_text = f"{num_students} student{'s' if num_students > 1 else ''}"
                input_text = f"Reserve seats in {subject} {course_num} section {class_suffix} for {student_text} with IDs {', '.join(student_ids)} until {expire_date}"
                
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE reservations SYSTEM "Reservations.dtd">\n<reservations campus="woebegon" year="2010" term="Fal" dateFormat="MM/dd/yyyy">\n'
                output_xml += f'    <reservation subject="{subject}" courseNbr="{course_num}" expire="{expire_date}" type="individual">\n'
                output_xml += f'        <class type="Lec" suffix="{class_suffix}"/>\n'
                
                for student_id in student_ids:
                    output_xml += f'        <student externalId="{student_id}"/>\n'
                
                output_xml += f'    </reservation>\n</reservations>'
                
            elif reservation_type == 'group':
                group_ext_id = random.choice(['G1', 'G2', 'G3'])
                group_code = random.choice(self.student_groups)
                expire_date = random.choice(["09/01/2010", "08/15/2010"])
                
                # Multiple classes
                num_classes = random.randint(1, 3)
                class_types = random.sample(['Lec', 'Lab', 'Rec'], num_classes)
                
                classes_text = ', '.join([f"{ct.lower()}" for ct in class_types])
                input_text = f"Reserve {limit} seats in {subject} {course_num} {classes_text} sections for {group_code.lower()} student group until {expire_date}"
                
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE reservations SYSTEM "Reservations.dtd">\n<reservations campus="woebegon" year="2010" term="Fal" dateFormat="MM/dd/yyyy">\n'
                output_xml += f'    <reservation subject="{subject}" courseNbr="{course_num}" limit="{limit}" expire="{expire_date}" type="group">\n'
                
                for i, ct in enumerate(class_types):
                    suffix = str(random.randint(1, 5))
                    output_xml += f'        <class type="{ct}" suffix="{suffix}"/>\n'
                
                output_xml += f'        <studentGroup externalId="{group_ext_id}" code="{group_code}"/>\n'
                output_xml += f'    </reservation>\n</reservations>'
                
            elif reservation_type == 'curriculum':
                # Academic area, classifications, and majors
                area = random.choice(self.academic_areas)
                classifications = random.sample(self.classifications[:4], random.randint(1, 2))
                major = random.choice(self.majors)
                
                # Multiple classes or no specific classes
                if random.random() < 0.6:
                    num_classes = random.randint(1, 3)
                    class_suffixes = [str(i) for i in range(1, num_classes + 1)]
                    classes_text = f" sections {', '.join(class_suffixes)}"
                    classes_xml = ''.join([f'        <class type="Lec" suffix="{suffix}"/>\n' for suffix in class_suffixes])
                else:
                    classes_text = ""
                    classes_xml = ""
                
                class_text = ', '.join(classifications)
                input_text = f"Reserve {limit} seats in {subject} {course_num}{classes_text} for {major} majors with {class_text} classification from academic area {area}"
                
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE reservations SYSTEM "Reservations.dtd">\n<reservations campus="woebegon" year="2010" term="Fal">\n'
                output_xml += f'    <reservation subject="{subject}" courseNbr="{course_num}" limit="{limit}" type="curriculum">\n'
                output_xml += classes_xml
                output_xml += f'        <academicArea externalId="{area}" abbreviation="{area}"/>\n'
                
                for classification in classifications:
                    output_xml += f'        <academicClassification externalId="{classification}" code="{classification}"/>\n'
                
                output_xml += f'        <major externalId="{major}" code="{major}"/>\n'
                output_xml += f'    </reservation>\n</reservations>'
                
            else:  # course reservation
                config_name = str(random.randint(1, 3))
                
                if random.random() < 0.7:
                    input_text = f"Reserve {limit} seats in {subject} {course_num} configuration {config_name}"
                    config_xml = f'        <configuration name="{config_name}"/>\n'
                else:
                    input_text = f"Reserve {limit} seats in {subject} {course_num} entire course"
                    config_xml = ""
                
                output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE reservations SYSTEM "Reservations.dtd">\n<reservations campus="woebegon" year="2010" term="Fal">\n'
                output_xml += f'    <reservation subject="{subject}" courseNbr="{course_num}" limit="{limit}" type="course">\n'
                output_xml += config_xml
                output_xml += f'    </reservation>\n</reservations>'
            
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
        elif days == 'TTh':
            return 'Tuesday Thursday'
        elif days == 'MW':
            return 'Monday Wednesday'
        elif days == 'WF':
            return 'Wednesday Friday'
        elif days == 'MTWRF':
            return 'Monday through Friday'
        elif days == 'MF':
            return 'Monday Friday'
        else:
            return ' '.join([day_map[d] for d in days if d in day_map])

    def day_to_readable(self, day):
        """Convert single day code to readable format"""
        day_map = {'M': 'Monday', 'T': 'Tuesday', 'W': 'Wednesday', 'R': 'Thursday', 'F': 'Friday'}
        return day_map.get(day, day)

    def level_to_text(self, level):
        """Convert preference level to readable text"""
        level_map = {
            'P': 'is prohibited from',
            'R': 'is required for',
            '-2': 'strongly dislikes',
            '-1': 'dislikes',
            '0': 'is neutral about',
            '1': 'prefers',
            '2': 'strongly prefers'
        }
        return level_map.get(level, 'has preference for')

    def generate_student_sectioning_data(self, num_samples=300):
        """Generate StudentSectioning.dtd training data"""
        data = []
        
        for _ in range(num_samples):
            student_id = str(random.randint(1001, 9999))
            num_courses = random.randint(3, 6)
            
            # Generate course requests
            course_requests = []
            for i in range(num_courses):
                subject = random.choice(self.subjects)
                course_num = random.choice(self.course_numbers)
                priority = i + 1
                alternative = random.random() < 0.2  # 20% are alternatives
                credit = random.choice([1, 2, 3, 4, 5])
                
                course_requests.append({
                    'subject': subject,
                    'courseNbr': course_num,
                    'priority': priority,
                    'alternative': alternative,
                    'credit': credit
                })
            
            # Natural language input
            primary_courses = [cr for cr in course_requests if not cr['alternative']]
            alt_courses = [cr for cr in course_requests if cr['alternative']]
            
            primary_text = ', '.join([f"{cr['subject']} {cr['courseNbr']}" for cr in primary_courses[:3]])
            input_text = f"Student {student_id} wants to enroll in {primary_text}"
            if alt_courses:
                alt_text = ', '.join([f"{cr['subject']} {cr['courseNbr']}" for cr in alt_courses[:2]])
                input_text += f" with alternatives {alt_text}"
            
            # XML output
            output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE students SYSTEM "StudentSectioning.dtd">\n<students campus="woebegon" year="2010" term="Fal">\n'
            output_xml += f'    <student externalId="{student_id}">\n'
            
            for cr in course_requests:
                alt_attr = ' alternative="true"' if cr['alternative'] else ''
                output_xml += f'        <courseRequest subject="{cr["subject"]}" courseNbr="{cr["courseNbr"]}" priority="{cr["priority"]}" credit="{cr["credit"]}"{alt_attr}/>\n'
            
            output_xml += f'    </student>\n</students>'
            
            data.append({
                "input": input_text,
                "output": output_xml,
                "type": "student_sectioning"
            })
            
        return data

    def generate_room_features_data(self, num_samples=200):
        """Generate RoomFeatures.dtd training data"""
        data = []
        features = ['Comp', 'Proj', 'Board', 'Video', 'Audio', 'Lab', 'Smart', 'Wifi']
        
        for _ in range(num_samples):
            building = random.choice(self.buildings)
            room = random.choice(self.room_numbers)
            capacity = random.choice([15, 20, 25, 30, 40, 50, 75, 100, 150])
            room_type = random.choice(['Classroom', 'Laboratory', 'Seminar', 'Lecture Hall'])
            
            # Assign features
            room_features = random.sample(features, random.randint(1, 4))
            
            features_text = ', '.join(room_features).lower()
            input_text = f"Room {building} {room} is a {room_type.lower()} with capacity {capacity} and features: {features_text}"
            
            # XML output
            output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE rooms SYSTEM "RoomFeatures.dtd">\n<rooms campus="woebegon">\n'
            output_xml += f'    <room building="{building}" roomNbr="{room}" capacity="{capacity}" type="{room_type}">\n'
            
            for feature in room_features:
                output_xml += f'        <feature name="{feature}"/>\n'
            
            output_xml += f'    </room>\n</rooms>'
            
            data.append({
                "input": input_text,
                "output": output_xml,
                "type": "room_features"
            })
            
        return data

    def generate_instructor_data(self, num_samples=150):
        """Generate InstructorInfo.dtd training data"""
        data = []
        
        for _ in range(num_samples):
            instructor = random.choice(self.instructors)
            dept = random.choice(self.departments)
            ext_id = str(random.randint(100, 999))
            position = random.choice(['Assistant Professor', 'Associate Professor', 'Professor', 'Lecturer', 'Adjunct'])
            max_load = random.choice([9, 12, 15, 18])
            
            # Teaching preferences
            preferred_subjects = random.sample(self.subjects, random.randint(1, 3))
            unavailable_days = random.sample(['M', 'T', 'W', 'R', 'F'], random.randint(0, 2))
            
            subjects_text = ', '.join(preferred_subjects)
            input_text = f"{instructor[0]} {instructor[2]} {instructor[1]} from department {dept} is {position.lower()} with {max_load} credit hour load, teaches {subjects_text}"
            
            if unavailable_days:
                days_text = ', '.join([self.day_to_readable(d) for d in unavailable_days])
                input_text += f" and unavailable on {days_text}"
            
            # XML output
            output_xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE instructors SYSTEM "InstructorInfo.dtd">\n<instructors campus="woebegon" year="2010" term="Fal">\n'
            output_xml += f'    <instructor externalId="{ext_id}" firstName="{instructor[2]}" lastName="{instructor[1]}" department="{dept}" position="{position}" maxLoad="{max_load}">\n'
            
            for subject in preferred_subjects:
                output_xml += f'        <teachingPreference subject="{subject}" level="2"/>\n'
            
            for day in unavailable_days:
                output_xml += f'        <unavailableDay day="{day}"/>\n'
            
            output_xml += f'    </instructor>\n</instructors>'
            
            data.append({
                "input": input_text,
                "output": output_xml,
                "type": "instructor_info"
            })
            
        return data

    def generate_complete_dataset(self, total_samples=2500):
        """Generate complete training dataset with all DTD types"""
        print("Generating UniTime XML training dataset...")
        
        all_data = []
        
        # Generate data for each DTD type
        print("Generating Course Offering data...")
        all_data.extend(self.generate_course_offering_data(int(total_samples * 0.18)))
        
        print("Generating Course Timetable data...")
        all_data.extend(self.generate_course_timetable_data(int(total_samples * 0.18)))
        
        print("Generating Preferences data...")
        all_data.extend(self.generate_preferences_data(int(total_samples * 0.18)))
        
        print("Generating Reservations data...")
        all_data.extend(self.generate_reservations_data(int(total_samples * 0.15)))
        
        print("Generating Student Sectioning data...")
        all_data.extend(self.generate_student_sectioning_data(int(total_samples * 0.11)))
        
        print("Generating Room Features data...")
        all_data.extend(self.generate_room_features_data(int(total_samples * 0.10)))
        
        print("Generating Instructor data...")
        all_data.extend(self.generate_instructor_data(int(total_samples * 0.10)))
        
        # Shuffle the dataset
        random.shuffle(all_data)
        
        print(f"Generated {len(all_data)} training samples")
        return all_data

    def save_dataset(self, data, filename="unitime_training_dataset.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filename}")

    def save_dataset_by_type(self, data, output_dir="unitime_datasets"):
        """Save dataset split by type"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Group by type
        by_type = {}
        for item in data:
            dtype = item['type']
            if dtype not in by_type:
                by_type[dtype] = []
            by_type[dtype].append(item)
        
        # Save each type separately
        for dtype, items in by_type.items():
            filename = os.path.join(output_dir, f"{dtype}_dataset.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(items)} {dtype} samples to {filename}")

    def export_for_fine_tuning(self, data, output_file="unitime_fine_tuning.jsonl"):
        """Export dataset in JSONL format for fine-tuning"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                training_example = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Convert this natural language description to UniTime XML format:\n{item['input']}"
                        },
                        {
                            "role": "assistant",
                            "content": item['output']
                        }
                    ]
                }
                f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
        print(f"Fine-tuning dataset exported to {output_file}")

# Example usage and main execution
# if __name__ == "__main__":
#     generator = EnhancedUniTimeDatasetGenerator()
    
#     # Generate complete dataset
#     dataset = generator.generate_complete_dataset(2500)
    
#     # Save in different formats
#     generator.save_dataset(dataset, "unitime_complete_dataset.json")
#     generator.save_dataset_by_type(dataset)
#     generator.export_for_fine_tuning(dataset)
    
#     # Print statistics
#     print("\nDataset Statistics:")
#     type_counts = {}
#     for item in dataset:
#         dtype = item['type']
#         type_counts[dtype] = type_counts.get(dtype, 0) + 1
    
#     for dtype, count in sorted(type_counts.items()):
#         print(f"  {dtype}: {count} samples")
    
#     print(f"\nTotal samples: {len(dataset)}")
#     print("Dataset generation complete!")
if __name__ == "__main__":
    generator = EnhancedUniTimeDatasetGenerator()
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset(3000)
    
    # Save the full dataset
    generator.save_dataset(dataset, "unitime_complete_dataset.json")
    generator.save_dataset_by_type(dataset)
    
    # Export full dataset for fine-tuning (JSONL format)
    generator.export_for_fine_tuning(dataset, output_file="unitime_fine_tuning.jsonl")
    
    # Split into train, val, test and save into data/processed/
    def split_dataset(data, train_ratio=0.8, val_ratio=0.1):
        random.shuffle(data)
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        return {
            "train": data[:train_end],
            "val": data[train_end:val_end],
            "test": data[val_end:]
        }

    splits = split_dataset(dataset)

    os.makedirs("data/processed", exist_ok=True)
    for split_name, split_data in splits.items():
        out_path = f"data/processed/{split_name}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"{split_name.capitalize()} split saved to {out_path} ({len(split_data)} samples)")

    # Dataset Statistics
    print("\nDataset Statistics:")
    type_counts = {}
    for item in dataset:
        dtype = item['type']
        type_counts[dtype] = type_counts.get(dtype, 0) + 1

    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype}: {count} samples")

    print(f"\nTotal samples: {len(dataset)}")
    print("Dataset generation complete!")