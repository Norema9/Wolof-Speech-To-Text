
from typing import Dict, Union
from utils.utils import process_text
import os
import random

class CreateDataset:
    def __init__(self,
                data_brut_file: Union[str, os.PathLike],
                punctuations: Union[str, os.PathLike],
                save_data_dir: Union[str, os.PathLike],
                train_text_file_name: str,
                train_label_file_name: str,
                val_text_file_name: str,
                val_label_file_name: str,
                test_text_file_name: str,
                test_label_file_name: str,
                punct_label_vocab_file: str,
                capit_label_vocab_file: str,
                chars_to_keep: str,
                char_to_replace_dict,
                special_tokens,
                train_ratio: float = 0.8,
                val_ratio: float = 0.1,
                test_ratio: float = 0.1,
                word_sep: str = "|"
                 ):
        self.data_brut_file = data_brut_file
        self.save_data_dir = save_data_dir
        self.train_text_file_name = train_text_file_name
        self.train_label_file_name = train_label_file_name
        self.val_text_file_name = val_text_file_name
        self.val_label_file_name = val_label_file_name
        self.test_text_file_name = test_text_file_name
        self.test_label_file_name = test_label_file_name
        self.punct_label_vocab_file = punct_label_vocab_file
        self.capit_label_vocab_file = capit_label_vocab_file
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.special_tokens = special_tokens
        self.word_sep = word_sep

        # Read lines from the data_brut_file
        with open(self.data_brut_file, 'r', encoding='utf-8') as f:
            self.data_lines = f.readlines()

        # Read characters from the char_to_keep_file
        self.chars_to_keep = chars_to_keep
        # Initialize a new list to store the updated characters to keep
        updated_chars_to_keep = list(self.chars_to_keep)

        # Iterate over each character in the original list
        for char in chars_to_keep:
            # Check if the character is a lowercase letter
            if char.islower():
                # Add the corresponding uppercase letter to the list
                updated_chars_to_keep.append(char.upper())

        # Convert the list to a string
        self.chars_to_keep = ''.join(updated_chars_to_keep)

        # Read punctuations from the punctuation_file
        self.punctuations = punctuations.strip()
        self.punctuations_labels_unique = set()
        for punct in self.punctuations:
            self.punctuations_labels_unique.add(punct)

        self.punctuations_labels_unique.add('O')
        self.punctuations_labels_unique = list(self.punctuations_labels_unique)
        self.capit_labels_unique = ['O', 'U']

        # Read characters to replace from the char_to_replace_file
        self.char_to_replace_dict = {key: value for key, value in char_to_replace_dict.items()}
        
        # Create the directory if it doesn't exist
        os.makedirs(self.save_data_dir, exist_ok=True)
    
    def create(self):
        self._clean()
        self.text_lines, self.label_lines = self._create_dataset()
        self._split_data()
        # self._check_lines()
        self._save_data()
        self._save_labels()

    def _clean(self) -> None:
        """
        Clean the dataset based on the loaded information.
        """
        # Clean data lines
        cleaned_data_lines = []
        for line in self.data_lines:
            cleaned_line = line
            # Replace characters based on char_to_replace_dict
            for key, value in self.char_to_replace_dict.items():
                cleaned_line = cleaned_line.replace(key, value)
            # Remove characters not in chars_to_keep
            cleaned_line = ''.join(c for c in cleaned_line if c in self.chars_to_keep or c in self.punctuations)
            cleaned_data_lines.append(cleaned_line)
        self.data_lines = cleaned_data_lines

    # TODO: Consider changing the shuffling so the relations between the lines is not lost. 
    # This could be done shuffling the data by paragrpah and not by lines
    def _split_data(self) -> None:
        """
        Split the dataset into train, validation, and test sets.
        """
        total_len = len(self.text_lines)
        train_len = int(total_len * self.train_ratio)
        val_len = int(total_len * self.val_ratio)

        # Shuffle the indices
        indices = list(range(total_len))
        random.shuffle(indices)

        # Split the indices
        self.train_indices = indices[:train_len]
        self.val_indices = indices[train_len:train_len + val_len]
        self.test_indices = indices[train_len + val_len:]

        # Split text lines
        self.train_text_lines = [self.text_lines[i] for i in self.train_indices]
        self.validation_text_lines = [self.text_lines[i] for i in self.val_indices]
        self.test_text_lines = [self.text_lines[i] for i in self.test_indices]

        # Split label lines
        self.train_label_lines = [self.label_lines[i] for i in self.train_indices]
        self.validation_label_lines = [self.label_lines[i] for i in self.val_indices]
        self.test_label_lines = [self.label_lines[i] for i in self.test_indices]

    def _create_dataset(self):
        text_lines, label_lines = process_text(self.data_lines, self.chars_to_keep.strip(), self.punctuations)
        return text_lines, label_lines

    def _contain_punctuation(self, word):
        for punct in self.punctuations:
            if punct in word:
                return True, punct
        return False, None
    
    def _save_labels(self):
        # Write the lines to the file
        filepath = os.path.join(self.save_data_dir, self.punct_label_vocab_file)
        with open(filepath, 'w', encoding='utf-8') as file:
            lines = "\n".join(self.punctuations_labels_unique)
            file.writelines(lines)
        # Write the lines to the file
        filepath = os.path.join(self.save_data_dir, self.capit_label_vocab_file)
        with open(filepath, 'w', encoding='utf-8') as file:
            lines = "\n".join(self.capit_labels_unique)
            file.writelines(lines)
 
    def _save_data(self):
        # Create the directory if it doesn't exist
        os.makedirs(self.save_data_dir, exist_ok=True)
        
        # Write the lines to the file
        filepath = os.path.join(self.save_data_dir, self.train_text_file_name)
        with open(filepath, 'w', encoding='utf-8') as file:
            lines = "\n".join(self.train_text_lines)
            file.writelines(lines)
        # Write the lines to the file
        filepath = os.path.join(self.save_data_dir, self.val_text_file_name)
        with open(filepath, 'w', encoding='utf-8') as file:
            lines = "\n".join(self.validation_text_lines)
            file.writelines(lines)
        # Write the lines to the file
        filepath = os.path.join(self.save_data_dir, self.test_text_file_name)
        with open(filepath, 'w', encoding='utf-8') as file:
            lines = "\n".join(self.test_text_lines)
            file.writelines(lines)


        # Write the lines to the file
        filepath = os.path.join(self.save_data_dir, self.train_label_file_name)
        with open(filepath, 'w', encoding='utf-8') as file:
            lines = "\n".join(self.train_label_lines)
            file.writelines(lines)
        # Write the lines to the file
        filepath = os.path.join(self.save_data_dir, self.val_label_file_name)
        with open(filepath, 'w', encoding='utf-8') as file:
            lines = "\n".join(self.validation_label_lines)
            file.writelines(lines)
        # Write the lines to the file
        filepath = os.path.join(self.save_data_dir, self.test_label_file_name)
        with open(filepath, 'w', encoding='utf-8') as file:
            lines = "\n".join(self.test_label_lines)
            file.writelines(lines)

    def _check_lines(self):
        for i, line in enumerate(self.train_label_lines):
            pairs = line.split()
            if not all([len(p) == 2 for p in pairs]):
                print(line)
                raise ValueError(
                    f"Some label pairs are not pairs but have wrong length (!= 2) in line {i} in label file"
                )
            words = self.train_text_lines[i].split()
            if len(pairs) != len(words):
                print(line)
                print(words)
                raise ValueError(
                    f"In line {i} in text file text_file number of words {len(words)} is not equal to the "
                    f"number of labels {len(pairs)} in labels file labels_file."
                )

    def get_vocab_dict(self) -> Dict[int, str]:
        all_text = " ".join(self.text_lines)
        for v in self.special_tokens.values():
            all_text = all_text.replace(v, '')
        vocab_list = list(set(all_text))
        vocab_list.sort()
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}

        vocab_dict[self.word_sep] = vocab_dict[" "]
        del vocab_dict[" "]
        for v in self.special_tokens.values():
            vocab_dict[v] = len(vocab_dict)
        print(vocab_dict)
        return vocab_dict