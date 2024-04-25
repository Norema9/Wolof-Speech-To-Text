import argparse
import os
import sys
import toml
os.chdir(r"E:\IA\WOLOF\LM")
sys.path.append(r"NGRAM\CODE")
from utils.utils import *



def main(data_brut_dir, data_clean_dir, characters, char_to_replace_dict):

    # Initialize a new list to store the updated characters to keep
    updated_chars_to_keep = list(characters)

    # Iterate over each character in the original list
    for char in characters:
        # Check if the character is a lowercase letter
        if char.islower():
            # Add the corresponding uppercase letter to the list
            updated_chars_to_keep.append(char.upper())

    # Convert the list to a string
    characters = ''.join(updated_chars_to_keep)

    txt_brut_files = list_txt_files(data_brut_dir)
    filepath = os.path.join(data_clean_dir, "data.txt")
    with open(filepath, 'w', encoding='utf-8') as cleaned_file:
        # lines = "\n".join(self.train_text_lines)
        for file in txt_brut_files:
            file_path = os.path.join(data_brut_dir, file)
            with open(file_path, 'r', encoding="utf-8") as f:
                f_text_lines = f.readlines()
                cleaned_text_lines = clean_text(f_text_lines, char_to_replace_dict, characters)
                processed_text_lines = process_text(cleaned_text_lines, characters)
                cleaned_file.writelines('\n'.join(processed_text_lines))
    

if __name__ == '__main__':
    data_brut_dir = r"E:\IA\WOLOF\LM\NGRAM\DATA\BRUT"
    data_clean_dir = r"E:\IA\WOLOF\LM\NGRAM\DATA\CLEANED"
    characters = "abcdefghijklmnopqrstuvwxyzñóŋéçèɓõẽáãë"
    char_to_replace_dict = {'ï' : 'a', 'î' : 'i', 'ā' : 'a', 'ƭ' : 'c', 'ī' : 'i',
                            'ä' : 'a', 'ɗ' : 'nd', 'ń' : 'ñ', 'ồ' : 'o', 'ї' : 'i',
                            'ü' : 'u', 'ù' : 'u', 'ú' : 'u', 'ă' : 'ã', 'â' : 'a',
                            'û' : 'u', 'è' : 'e', 'ç' : 's', 'ö' : 'o', 'ý' : 'y',
                            'ì' : 'i', 'í' : 'i', 'ɓ' : 'b', 'ô' : 'o', 'ê' : 'e',
                            'à' : 'a'}

    main(data_brut_dir, data_clean_dir, characters, char_to_replace_dict)

