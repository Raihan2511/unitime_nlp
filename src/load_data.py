# load_data.py
import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_datasets():
    base_path = '/home/sysadm/Music/unitime_nlp/data'
    offerings = load_json(f'{base_path}/offerings_data/train_offer.json')
    reservations = load_json(f'{base_path}/reservation_data/train_reservation.json')
    preferences = load_json(f'{base_path}/preference_data/train_pref.json')
    return offerings, reservations, preferences
