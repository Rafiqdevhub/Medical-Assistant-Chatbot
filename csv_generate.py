import csv
import logging
import os
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
REQUIRED_COLUMNS = ["Symptom", "Possible Condition", "Suggested Treatment"]
FILE_NAME = "medical_symptoms_treatments.csv"
BACKUP_DIR = "backups"

def validate_data_entry(entry: List[str], columns: List[str]) -> bool:
    """Validate a single data entry"""
    if len(entry) != len(columns):
        return False
    return all(isinstance(item, str) and item.strip() for item in entry)

def create_backup(file_path: str) -> None:
    """Create a backup of the existing CSV file if it exists"""
    try:
        if os.path.exists(file_path):
            backup_dir = Path(BACKUP_DIR)
            backup_dir.mkdir(exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{Path(file_path).stem}_{timestamp}.csv"
            
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
    except Exception as e:
        logger.warning(f"Failed to create backup: {str(e)}")

def generate_medical_data() -> List[List[str]]:
    """Generate medical data with validation"""
    data = [
        REQUIRED_COLUMNS,
        ["Fever", "Viral Infection", "Rest, fluids, paracetamol"],
        ["Headache", "Migraine", "Pain relievers, rest, avoid triggers"],
        ["Cough", "Common Cold", "Honey, warm fluids, cough syrup"],
        ["Sore Throat", "Strep Throat", "Gargle with salt water, antibiotics"],
        ["Shortness of Breath", "Asthma", "Inhaler, avoid allergens"],
        ["Stomach Pain", "Gastritis", "Antacids, dietary changes"],
        ["Diarrhea", "Food Poisoning", "Hydration, probiotics"],
        ["Chest Pain", "Heart Disease", "Seek immediate medical attention"],
        ["Fatigue", "Anemia", "Iron supplements, healthy diet"],
        ["Joint Pain", "Arthritis", "Pain relievers, exercise"],
        ["Nausea", "Motion Sickness", "Anti-nausea medication, fresh air"],
        ["Dizziness", "Low Blood Pressure", "Hydration, rest, medical consultation"],
        ["Rash", "Allergic Reaction", "Antihistamines, avoid allergens"],
        ["Back Pain", "Muscle Strain", "Rest, gentle stretching, pain relievers"]
    ]
    return data

def write_csv(data: List[List[str]], file_path: str) -> None:
    """Write data to CSV with error handling"""
    try:
        # Validate all entries before writing
        headers = data[0]
        for i, entry in enumerate(data[1:], 1):
            if not validate_data_entry(entry, headers):
                raise ValueError(f"Invalid data entry at row {i}: {entry}")
        
        # Create backup of existing file
        create_backup(file_path)
        
        # Write new data
        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(data)
            
        logger.info(f"Successfully created CSV file: {file_path}")
        
        # Verify file was written correctly
        verify_file_contents(file_path, len(data))
        
    except Exception as e:
        logger.error(f"Error writing CSV file: {str(e)}")
        raise

def verify_file_contents(file_path: str, expected_rows: int) -> None:
    """Verify the written file contents"""
    try:
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            actual_rows = sum(1 for _ in reader)
            
        if actual_rows != expected_rows:
            raise ValueError(f"File verification failed: Expected {expected_rows} rows, got {actual_rows}")
        
        logger.info("File contents verified successfully")
            
    except Exception as e:
        logger.error(f"File verification failed: {str(e)}")
        raise

def main():
    try:
        # Generate medical data
        data = generate_medical_data()
        
        # Write to CSV
        write_csv(data, FILE_NAME)
        
    except Exception as e:
        logger.error(f"Failed to generate CSV: {str(e)}")
        raise

if __name__ == "__main__":
    main()
