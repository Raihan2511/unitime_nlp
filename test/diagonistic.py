import json
import os

def diagnose_jsonl_file(file_path):
    """
    Diagnose issues with a JSONL file
    """
    print(f"Diagnosing file: {file_path}")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print("❌ File does not exist!")
        return
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"📁 File size: {file_size} bytes")
    
    if file_size == 0:
        print("❌ File is empty!")
        return
    
    # Read and analyze the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    print(f"📄 Total lines in file: {len(lines)}")
    
    # Check each line
    valid_json_count = 0
    empty_lines = 0
    errors = []
    
    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        if not line_stripped:
            empty_lines += 1
            continue
            
        try:
            json.loads(line_stripped)
            valid_json_count += 1
        except json.JSONDecodeError as e:
            errors.append({
                'line': line_num,
                'error': str(e),
                'content_preview': line_stripped[:100] + ('...' if len(line_stripped) > 100 else '')
            })
    
    print(f"✅ Valid JSON lines: {valid_json_count}")
    print(f"📝 Empty lines: {empty_lines}")
    print(f"❌ Lines with JSON errors: {len(errors)}")
    
    if errors:
        print("\n🔍 First few errors:")
        for error in errors[:3]:
            print(f"  Line {error['line']}: {error['error']}")
            print(f"    Content: {error['content_preview']}")
            print()
    
    # Show first few lines of the file
    print("\n📋 First 3 lines of the file:")
    for i, line in enumerate(lines[:3], 1):
        print(f"Line {i}: {repr(line[:200])}")
    
    # Check if lines are properly separated
    print(f"\n🔗 Line endings: {repr(lines[0][-10:]) if lines else 'No lines'}")
    
    return valid_json_count, len(errors)

def test_json_loading(file_path, max_lines=5):
    """
    Test loading JSON lines one by one
    """
    print(f"\n🧪 Testing JSON loading from: {file_path}")
    print("-" * 30)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > max_lines:
                    break
                    
                line = line.strip()
                if not line:
                    print(f"Line {line_num}: Empty line (skipped)")
                    continue
                
                try:
                    data = json.loads(line)
                    print(f"Line {line_num}: ✅ Valid JSON")
                    
                    # Check structure
                    if isinstance(data, dict):
                        keys = list(data.keys())
                        print(f"  Keys: {keys[:5]}{'...' if len(keys) > 5 else ''}")
                    
                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: ❌ JSON Error: {e}")
                    print(f"  Content preview: {line[:100]}...")
                    
    except Exception as e:
        print(f"❌ Error opening file: {e}")

def check_file_content_structure(file_path):
    """
    Check the actual content structure of the file
    """
    print(f"\n🔍 Analyzing file content structure...")
    
    try:
        with open(file_path, 'rb') as f:
            first_1000_bytes = f.read(1000)
        
        # Check for common issues
        content_str = first_1000_bytes.decode('utf-8', errors='ignore')
        
        print(f"First 200 characters:")
        print(repr(content_str[:200]))
        
        # Count braces to see if JSON objects are properly separated
        open_braces = content_str.count('{')
        close_braces = content_str.count('}')
        newlines = content_str.count('\n')
        
        print(f"\nCharacter counts in first 1000 bytes:")
        print(f"  Opening braces: {open_braces}")
        print(f"  Closing braces: {close_braces}")
        print(f"  Newlines: {newlines}")
        
        # Check if there are consecutive closing/opening braces (bad format)
        if '}{' in content_str:
            print("⚠️  Found '}{' pattern - this indicates improperly separated JSON objects!")
            positions = [i for i in range(len(content_str)) if content_str[i:i+2] == '}{']
            print(f"  Found at positions: {positions[:5]}")
        
    except Exception as e:
        print(f"❌ Error analyzing file content: {e}")

# Main diagnostic function
def full_diagnosis(file_path):
    """
    Run full diagnosis on the JSONL file
    """
    print("🔬 JSONL File Diagnosis")
    print("=" * 50)
    
    # Basic file analysis
    valid_count, error_count = diagnose_jsonl_file(file_path)
    
    # Content structure analysis
    check_file_content_structure(file_path)
    
    # Test JSON loading
    test_json_loading(file_path)
    
    # Provide recommendations
    print("\n💡 Recommendations:")
    if error_count > 0:
        print("1. Your file has JSON parsing errors - it's not in proper JSONL format")
        print("2. Use the fix_jsonl_format script to repair the file")
        print("3. Each line should contain exactly one complete JSON object")
    elif valid_count == 0:
        print("1. No valid JSON objects found")
        print("2. Check if the file is corrupted or in a different format")
    else:
        print("1. File appears to be in correct JSONL format")
        print("2. The error might be in your loading code")

if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "/home/sysadm/Music/unitime_nlp/unitime_production_dataset.jsonl"
    full_diagnosis(file_path)