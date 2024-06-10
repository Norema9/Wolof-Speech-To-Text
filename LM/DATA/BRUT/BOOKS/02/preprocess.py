import os

os.chdir(r"C:\Users\maron\OneDrive\02-Documents\03.PROJETS\00.INFORMATIQUE\02.AI\WOLOF\LM\DATA\BRUT\02")

with open('cleaned_text.txt', mode='w', encoding='utf-8') as file_insert:
    with open('text.txt', mode='r', encoding='utf-8') as file:
        exception = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        new_line = ''
        restart = 1
        for line in file:
            for c in line.split(' '):
                for nbre in exception:
                    if nbre in c:
                        s =''
                        for i in new_line.split('\n'):
                            s = s + i
                        if s != '':
                            file_insert.write(s + '\n')
                        new_line = ''
                        restart = 1
                if restart == 0 and c != '\n':
                    if new_line == '':
                        new_line = c
                    else:
                        new_line = new_line + ' ' + c
                elif new_line == '' and restart == 1:
                    restart = 0