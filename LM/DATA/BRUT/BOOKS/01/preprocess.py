import os

os.chdir(r"C:\Users\maron\OneDrive\02-Documents\03.PROJETS\00.INFORMATIQUE\02.AI\WOLOF\LM\DATA\BRUT\01")

with open('cleaned_text.txt', mode='w', encoding='utf-8') as file_insert:
    with open('text.txt', mode='r', encoding='utf-8') as file:
        i = 0
        punctuation = ['.']
        exception = ['Seex', 'Muusaa', 'Ka', 'Ka,', 'Sëriñam.', 'Xadiim', 'Seexul', 'Yàlla', 'Yàlla,', 'Mustafaa', 'Bàmba', 'Aziim', 'Laahal']
        new_line = ''
        for line in file:
            new_line = ''
            if i > 7:
                for c in line.split(' '):
                    if c[0].isupper() and not c in exception:
                        new_line = new_line + '\n' + c
                    else:
                        new_line = new_line + ' ' + c
            else:
                new_line = line
            if new_line != '' and new_line != '\n':
                print(new_line)
                file_insert.write(new_line)
            i += 1