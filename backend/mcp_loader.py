import json
from pathlib import Path
import requests

def load_and_clean_cuny_json(output_path="data/cleaned_programs.json", year_filter=None):
    """
    Load CUNY program data from NY.gov API and clean it for use.
    
    Args:
        output_path (str): Path where cleaned data will be saved
        year_filter (str): Optional year to filter enrollment periods (e.g., "2024", "2025")
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("🔄 Fetching CUNY program data from NY.gov API...")
        
        # Build URL with optional year filter
        base_url = "https://data.ny.gov/resource/28gk-bu58.json"
        params = {"$limit": "50000"}
        
        if year_filter:
            # Filter for periods containing the specified year
            params["$where"] = f"period like '%{year_filter}%'"
            print(f"📅 Filtering for enrollment periods containing '{year_filter}'")
        
        # Construct URL with parameters
        url = base_url + "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for any  bad responses
        
        data = response.json()
        print(f"📥 Retrieved {len(data)} raw records")
        
        if not data:
            print("⚠️ No data received from API")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data from API: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON response: {e}")
        return False
    
    cleaned = []
    skipped_count = 0

    for item in data:
        if "program_name" not in item or "college" not in item:
            skipped_count += 1
            continue

        cleaned.append({
            "program_name": item.get("program_name", "").title(), #college major
            "college": item.get("college", "").title(), #CUNY campuses
            "degree_type": item.get("award_name", "").upper(),  #level of degree like (BS, MBA, phd ...etc)
            "date_program_established": item.get("date_first_registered", ""),  #When the program was established
            "irp_code": item.get("irp_code", ""),  # The Institution Registration Program code (What NY state uses for internal tracking)
            "cip_code": item.get("cip_2020_code", ""),  # This is the Classification of Instructional Programs code (federal standard for tracking)
            "cip_title": item.get("cip_2020_title_short", ""),  # Degree discription (Think Computer Science, Accounting ...etc)
            "tap_eligible": item.get("tap_eligible", ""),  # Does this program qualify for NYS TAP (yes / no)
            "record_type": item.get("record_type", ""),  # Type of record (Enrollment/Graduation)
            "period": item.get("period", "")  # The Current enrollment period("fall_2024")
        })
    
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} records missing required fields")
    else:
        print(f"👍 No Skipped records {skipped_count}")
    
    try:
        # Create the data folder if it doesnt already exist
        Path("data").mkdir(exist_ok=True)
        
        # This saves the cleaned json data
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        
        print(f"✔️ Saved {len(cleaned)} cleaned records to {output_path}")
        return True
        
    except IOError as e:
        print(f"❌ Error saving file: {e}")
        return False

if __name__ == "__main__":
    # REMEMBER!!! --> cofiguration: pls change this year for the most up to date information
    TARGET_YEAR = "2024"  # 2025 data not available in the NY.gov
    
    print(f" 🎯Target enrollment period for: {TARGET_YEAR}")
    success = load_and_clean_cuny_json(year_filter=TARGET_YEAR)
    
    if not success:
        print("❌ Failed to load data")
        exit(1)
    else:
        print(f"Loaded {TARGET_YEAR} enrollment data!😎😎😎")