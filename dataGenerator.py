from data_generator.preference import generate_preference_dataset
from data_generator.reservations import generate_reservations_dataset

print("\nGenerating preference dataset...\n")
generate_preference_dataset()
print("\nPreference dataset generation complete.\n")
generate_reservations_dataset()
print("Reservation dataset generation complete.")