# load_data.py
import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_train_datasets():
    base_path = '/home/sysadm/Music/unitime_nlp/data'
    offerings_train = load_json(f'{base_path}/offerings_data/train_offer.json')
    reservations_train = load_json(f'{base_path}/reservation_data/train_reservation.json')
    preferences_train = load_json(f'{base_path}/preference_data/train_pref.json')
    timetable_train = load_json(f'{base_path}/course_timetable/train_timetable.json')
    return offerings_train, reservations_train, preferences_train, timetable_train
def load_val_datasets():
    base_path = '/home/sysadm/Music/unitime_nlp/data'
    offerings_val = load_json(f'{base_path}/offerings_data/val_offer.json')
    reservations_val = load_json(f'{base_path}/reservation_data/val_reservation.json')
    preferences_val = load_json(f'{base_path}/preference_data/val_pref.json')
    timetable_val = load_json(f'{base_path}/course_timetable/val_timetable.json')
    return offerings_val, reservations_val, preferences_val, timetable_val  
def load_test_datasets():
    base_path = '/home/sysadm/Music/unitime_nlp/data'
    offerings_test = load_json(f'{base_path}/offerings_data/test_offer.json')
    reservations_test = load_json(f'{base_path}/reservation_data/test_reservation.json')
    preferences_test = load_json(f'{base_path}/preference_data/test_pref.json')
    timetable_test = load_json(f'{base_path}/course_timetable/test_timetable.json')
    return offerings_test, reservations_test, preferences_test, timetable_test
