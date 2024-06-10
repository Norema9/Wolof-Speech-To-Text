import os
import re

os.chdir(r"C:\Users\maron\OneDrive\02-Documents\03.PROJETS\00.INFORMATIQUE\02.AI\WOLOF\LM\DATA\BRUT\04")
characters = ["?", ".", "!", ":"]
auth = ["Mth", "Aoa", "Glt", "Eph", "Phl", "1Th", "Hbr", "1Pt", "2Pt", "1Jh", "Jde", "Rvl", "Mrk", "Lke", "Jhn", "Rmn", "1Cr", "2Cr", "1Tm", "Cls", "2Th", "2Tm", "Tts", "Phm", "Jms"]
# Join the characters with the pipe '|' to create the pattern for re.split()
pattern = "|".join(map(re.escape, characters))

chars_to_ignore_regex = r'[,\[\]?\.\!\-\;:"»“«%\‘”�\r\n*/\(\)0-9]'

with open('cleaned_text.txt', mode='w', encoding='utf-8') as file_insert:
    with open('text.txt', mode='r', encoding='utf-8') as file:
        new_line = ''
        start = 0 
        for line in file:
            if len(line) > len("Page"):
                if line[:4] != "Page":
                    start = 0
                    s = ""
                    for l in line.split('\n'):
                        s = s + l
                    # Use re.split() to split the text
                    sentences = re.split(pattern, s)
                    for sentence in sentences:
                        sentence_clean = ""
                        if sentence[:3] in auth:
                            for a in sentence.split(' ')[3:]:
                                if sentence_clean == '':
                                    sentence_clean = a
                                else:
                                    sentence_clean = sentence_clean + ' ' + a
                            start = 1
                        else:
                            sentence_clean = sentence

                        s = ""
                        for l in sentence_clean.split('\n'):
                            s = s + l
                        sentence_clean = s
                        if sentence_clean != '':
                            sent = re.sub(chars_to_ignore_regex, '', sentence_clean).lower()
                            if start == 0:
                                new_line = new_line + ' ' + sent.rstrip().lstrip()
                            else:
                                if new_line != '':
                                    if sent != '' and len(sent) > 1:
                                        new_line = new_line + '\n' + sent.rstrip().lstrip()
                                else:
                                    new_line = new_line + ' ' + sent.rstrip().lstrip()
                            start = 1

        file_insert.write(new_line)