import os
import re

os.chdir(r"C:\Users\maron\OneDrive\02-Documents\03.PROJETS\00.INFORMATIQUE\02.AI\WOLOF\LM\DATA\BRUT\06")
characters = ["?", ".", "!", ":"]

# Join the characters with the pipe '|' to create the pattern for re.split()
pattern = "|".join(map(re.escape, characters))
chars_to_ignore_regex = r'[,\{\}\[\]?\.\!\-\;:"»“«%\‘”�\r\n*/\(\)0-9]'

with open('cleaned_text.txt', mode='w', encoding='utf-8') as file_insert:
    with open('text.txt', mode='r', encoding='utf-8') as file:
        new_line = ''
        start = 0 
        for line in file:
            start = 0
            s = ""
            for l in line.split('\n'):
                s = s + l
            # Use re.split() to split the text
            sentences = re.split(pattern, s)
            for sentence in sentences:
                sent = re.sub(chars_to_ignore_regex, '', sentence).lower()
                if start == 0:
                    new_line = new_line + sent.rstrip().lstrip()
                else:
                    if sent != '' and len(sent) > 1:
                        new_line = new_line + '\n' + sent.rstrip().lstrip()
                start = 1
        file_insert.write(new_line)