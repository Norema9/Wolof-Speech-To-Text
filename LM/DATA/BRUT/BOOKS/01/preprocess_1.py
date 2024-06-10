import os

os.chdir(r"C:\Users\maron\OneDrive\02-Documents\03.PROJETS\00.INFORMATIQUE\02.AI\WOLOF\LM\DATA\BRUT\01")

with open('cleaned_text_1.txt', mode='w', encoding='utf-8') as file_insert:
    with open('cleaned_text.txt', mode='r', encoding='utf-8') as file:
        for line in file:
            if line and line != '\n' and line != '':
                file_insert.write(line)