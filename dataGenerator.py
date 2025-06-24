from data_generator.preference import generate_preference_dataset
from data_generator.reservations import generate_reservations_dataset
from data_generator.offerings import generate_offerings_dataset
from data_generator.course_timetable import generate_timetable_dataset

print("\nGenerating preference dataset...\n")
generate_preference_dataset()
print("\nPreference dataset generation complete.\n")
generate_reservations_dataset()
print("Reservation dataset generation complete.")
print("\ncreate offerings dataset...\n")
generate_offerings_dataset()
print("\nOfferings dataset generation complete.\n")
print("\nCreating course timetable dataset...\n")
generate_timetable_dataset()
print("\nCourse timetable dataset generation complete.\n")
