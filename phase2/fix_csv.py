"""
Script to fix the corrupted CSV file by reading it with error handling
and re-writing it with proper quoting and escaping.
"""
import pandas as pd
import csv
import sys

INPUT_FILE = "../outputs/advbench_suffixes_all_models.csv"
OUTPUT_FILE = "../outputs/advbench_suffixes_all_models_fixed.csv"
BACKUP_FILE = "../outputs/advbench_suffixes_all_models_backup.csv"

def fix_csv():
    """
    Reads the corrupted CSV file line by line with error handling,
    then writes a properly formatted version.
    """
    import shutil

    print(f"[*] Backing up original file to {BACKUP_FILE}...")
    try:
        shutil.copy(INPUT_FILE, BACKUP_FILE)
        print(f"[+] Backup created successfully")
    except Exception as e:
        print(f"[-] Failed to create backup: {e}")
        return False

    print(f"[*] Reading corrupted CSV with error handling...")

    # Try different approaches to read the file
    rows = []
    header = None

    # Approach 1: Read with on_bad_lines='skip'
    try:
        df = pd.read_csv(INPUT_FILE, on_bad_lines='skip', engine='python')
        print(f"[+] Successfully read {len(df)} rows using on_bad_lines='skip'")

        # Write the fixed version
        print(f"[*] Writing fixed CSV to {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"[+] Fixed CSV written successfully")

        # Replace original with fixed version
        print(f"[*] Replacing original file with fixed version...")
        shutil.copy(OUTPUT_FILE, INPUT_FILE)
        print(f"[+] Original file replaced successfully")

        print(f"\n[+] CSV file fixed! {len(df)} rows recovered.")
        print(f"    Original backed up to: {BACKUP_FILE}")
        print(f"    Fixed version saved to: {INPUT_FILE}")
        return True

    except Exception as e:
        print(f"[-] Failed to read CSV: {e}")

        # Approach 2: Read line by line manually
        print(f"[*] Attempting manual line-by-line recovery...")
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                header = lines[0].strip()
                print(f"[+] Header: {header}")

                # Parse header
                import io
                header_fields = list(csv.reader([header], quotechar='"', escapechar='\\'))[0]
                print(f"[+] Found {len(header_fields)} columns: {header_fields}")

                # Try to parse each line
                for i, line in enumerate(lines[1:], start=2):
                    try:
                        # Parse with csv reader
                        parsed = list(csv.reader([line.strip()], quotechar='"', escapechar='\\'))[0]
                        if len(parsed) == len(header_fields):
                            rows.append(parsed)
                        else:
                            print(f"[!] Line {i}: Expected {len(header_fields)} fields, got {len(parsed)} - skipping")
                    except Exception as e:
                        print(f"[!] Line {i}: Parse error - {e} - skipping")

                if rows:
                    # Create DataFrame
                    df = pd.DataFrame(rows, columns=header_fields)
                    print(f"[+] Recovered {len(df)} valid rows out of {len(lines)-1} total")

                    # Write the fixed version
                    print(f"[*] Writing fixed CSV to {OUTPUT_FILE}...")
                    df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
                    print(f"[+] Fixed CSV written successfully")

                    # Replace original with fixed version
                    print(f"[*] Replacing original file with fixed version...")
                    shutil.copy(OUTPUT_FILE, INPUT_FILE)
                    print(f"[+] Original file replaced successfully")

                    print(f"\n[+] CSV file fixed! {len(df)} rows recovered.")
                    print(f"    Original backed up to: {BACKUP_FILE}")
                    print(f"    Fixed version saved to: {INPUT_FILE}")
                    return True
                else:
                    print(f"[-] No valid rows could be recovered")
                    return False

        except Exception as e:
            print(f"[-] Manual recovery failed: {e}")
            return False

if __name__ == "__main__":
    print("="*60)
    print("CSV Fix Utility")
    print("="*60)

    success = fix_csv()

    if success:
        print(f"\n[+] Success! You can now run: python advers_attack.py")
        sys.exit(0)
    else:
        print(f"\n[-] Failed to fix CSV. You may need to regenerate the data.")
        sys.exit(1)

