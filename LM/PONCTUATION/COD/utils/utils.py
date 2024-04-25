import importlib
import torch
import numpy as np
import random
import os
import re

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def initialize_module(path: str, args: dict = None, initialize: bool = True):
    module_path = ".".join(path.split(".")[:-1])
    class_or_function_name = path.split(".")[-1]

    module = importlib.import_module(module_path)
    class_or_function = getattr(module, class_or_function_name)

    if initialize:
        if args:
            return class_or_function(**args)
        else:
            return class_or_function()
    else:
        return class_or_function


def load_text_file(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []


class CustomError(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

def process_text(text_brut_lines, characters, punctuation):
    text_lines = []
    label_lines = []

    for i, line in enumerate(text_brut_lines):
        text_line = []
        label_line = []

        words = re.findall(r'\b[' + characters + r']+\b|[' + punctuation + r']', line)

        for word in words:
            if word not in punctuation:
                if word[0].isupper():
                    case_label = 'U'
                else:
                    case_label = 'O'

                if re.match(r'\b[' + characters + r']+\b', word):
                    punct_label = 'O'
                else:
                    punct_label = word

                word = word.lower()

                text_line.append(word)
                label_line.append((punct_label + case_label))
            else:
                if len(label_line) > 0:
                    prev_label = label_line[-1]
                    label_line[-1] = word + prev_label[1]
                else:
                    label_lines[-1] = label_lines[-1][:-2] + word + label_lines[-1][-1]
    
        if len(text_line) > 0 and len(label_line) > 0 and len(text_line) == len(label_line):
            text_lines.append(' '.join(text_line))
            label_lines.append(' '.join(label_line))
    if len(text_lines) == len(label_lines):
        return text_lines, label_lines
    else:
       raise ValueError(f'the text data and the label data have different length') 

