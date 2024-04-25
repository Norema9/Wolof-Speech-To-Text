import re
import os

# List all mp3 files in a directory
def list_txt_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.txt')]

def clean_text(data_lines:list, char_to_replace_dict: dict, characters:str):
    """
    Clean the dataset based on the loaded information.
    """
    # Clean data lines
    cleaned_data_lines = []
    for line in data_lines:
        cleaned_line:str = line
        # Replace characters based on char_to_replace_dict
        for key, value in char_to_replace_dict.items():
            cleaned_line = cleaned_line.replace(key, value)
        # Remove characters not in chars_to_keep
        cleaned_line = ''.join(c for c in cleaned_line if c in characters or c in " ")
        cleaned_data_lines.append(cleaned_line)
    return cleaned_data_lines

def load_text_file(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []

def process_text(text_brut_lines, characters):
    text_lines = []

    for line in text_brut_lines:
        text_line = []

        # Modify the regex pattern
        pattern = '[' + characters + ']+'

        # Find all matches using the modified pattern
        words = re.findall(pattern, line)

        for word in words:
            word = word.lower()
            text_line.append(word)
                
        text_lines.append(' '.join(text_line))
    return text_lines

