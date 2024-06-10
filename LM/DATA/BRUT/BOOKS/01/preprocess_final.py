import os
import re

os.chdir(r"C:\Users\maron\OneDrive\02-Documents\03.PROJETS\00.INFORMATIQUE\02.AI\WOLOF\LM\DATA\BRUT\01")
characters = ["?", ".", "!", ":"]

# Join the characters with the pipe '|' to create the pattern for re.split()
pattern = "|".join(map(re.escape, characters))
chars_to_ignore_regex = r'[,\{\}\[\]?\.\!\-\;:"»“«%\‘”�\r\n*/\(\)0-9]'

with open('cleaned_text_last.txt', mode='w', encoding='utf-8') as file_insert:
    with open('cleaned_text_last_1.txt', mode='r', encoding='utf-8') as file:
        new_line = ''
        for line in file:
            s = ""
            for l in line.split('\n'):
                s = s + l
            sent = re.sub(chars_to_ignore_regex, '', s).lower()
            if sent != '' and len(sent) > 1:
                new_line = new_line + '\n' + sent.rstrip().lstrip()
      
        file_insert.write(new_line)