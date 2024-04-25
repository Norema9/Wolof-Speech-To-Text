import pandas as pd
import sys
import os
import re
import regex
import librosa
import numpy as np
from pandarallel import pandarallel
from typing import Dict, List

# For testing 
sys.path.append('..')

from sklearn.model_selection import train_test_split
from utils.feature import load_wav
from tqdm import tqdm
from torch.utils.data import Dataset
from dataloader.dataset import Dataset as InstanceDataset


class BaseDataset(Dataset):
    def __init__(self, rank, dist, path, sr, delimiter, special_tokens, min_duration = -np.inf, max_duration = np.inf, preload_data = False, transform = None, nb_workers = 4):
        self.rank = rank
        self.dist = dist
        self.sr = sr
        # Special characters to remove in your data 
        self.chars_to_ignore = r'[,?.!\-;:"“%\'�]'
        self.chars_to_ignore = r'[кɲớˈ\'\xa0\r\n]'
        self.chars_to_keep = r'[^a-zA-Z\sёñïóŋöäàéîā́сđớ\'ˈоɗɲtx їüúaëçèĩã̈ûjämсукéеɓìs️öŋïõăаrýànóvñlò̃qẽyfƭhgziâwíńồpêáôùёībkр]'
        self.replace_dict = {'ï': 'a', 'î': 'i', 'ā': 'a', 'ƭ': 'c', 'ī': 'i', 'ä': 'a', 'ɗ': 'nd', 'ń': 'ñ', 'ồ': 'o',
                    'ї': 'i', 'ü': 'u', 'ù': 'u', 'ú': 'u', 'ă': 'ã', '̃': '', 'â': 'a', '́': '', 'û': 'u',
                    '̈': '', 'è': 'e', 'ç': 's', 'ö': 'o', 'ý': 'y', 'ì': 'i', 'í': 'i', '̀': '', 'ɓ':'b', 'ô':'o',
                    'ê':'e', 'à':'a'}
        self.transform = transform
        self.preload_data = preload_data
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.df = self.load_data(path, delimiter)
        self.special_tokens = special_tokens
        pandarallel.initialize(progress_bar=True, nb_workers = nb_workers)

        if min_duration != -np.inf or max_duration != np.inf:
            if self.rank == 0 and 'duration' not in self.df.columns:
                print("\n*****Generate duration column*****")
                self.df['duration'] = self.df['path'].parallel_apply(lambda filename: librosa.get_duration(filename=filename))
                self.df.to_csv(path, index = False, sep = delimiter)
            self.dist.barrier()
            self.df = self.load_data(path, delimiter)
            if self.rank == 0:
                print("\n*****Filter out invalid audio*****")
            mask = (self.df['duration'] <= self.max_duration) & (self.df['duration'] >= self.min_duration)
            self.df = self.df[mask]
        # self.df['transcript'] = self.df['transcription'].parallel_apply(self.remove_special_characters)
        self.df['transcript'] = self.df['transcription'].apply(self.remove_special_characters)
    
        if self.preload_data:
            if self.rank == 0:
                print(f"\n*****Preloading {len(self.df)} data*****")
            self.df['wav'] = self.df['path'].parallel_apply(lambda filepath: load_wav(filepath, sr = self.sr))
    
    def remove_special_characters(self, transcript) -> str:
        cleaned_transcript = ""
        for word in transcript.split(','):
            if cleaned_transcript != "":
                cleaned_transcript = cleaned_transcript + ", " + word.rstrip().lstrip()
            else:
                cleaned_transcript = word
        cleaned_transcript = regex.sub(self.chars_to_keep, '', cleaned_transcript).lower() + " "
        cleaned_transcript = regex.sub(self.chars_to_ignore, '', cleaned_transcript) + " "

        for key, value in self.replace_dict.items():
            cleaned_transcript = regex.sub(key, value, cleaned_transcript) + " "
        return cleaned_transcript

    def get_vocab_dict(self) -> Dict[int, str]:
        # Read https://huggingface.co/blog/fine-tune-wav2vec2-english for more information
        all_text = " ".join(list(self.df["transcript"]))
        #  remove special tokens in all_text, otherwise it will tokenize the special tokens' characters. Eg: <unk> -> '<', 'u', 'n', 'k', '>'
        for v in self.special_tokens.values():
            all_text = all_text.replace(v, '')
        vocab_list = list(set(all_text))
        vocab_list.sort()
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}

        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        for v in self.special_tokens.values():
            vocab_dict[v] = len(vocab_dict)
        print(vocab_dict)
        return vocab_dict

    def preload_dataset(self, paths, sr) -> List:
        wavs = []
        print("Preloading {} data".format(self.mode))
        for path in tqdm(paths, total = len(paths)):
            wav = load_wav(path, sr)
            wavs += [wav]
        return wavs

    def load_data(self, path, delimiter) -> pd.DataFrame:
        df = pd.read_csv(path, delimiter = delimiter)
        return df

    def get_data(self) -> Dataset:
        ds = InstanceDataset(self.df, self.sr, self.preload_data, self.transform)
        return ds


if __name__ == '__main__':
    ds = BaseDataset(
        path = r'SPEECH_TO_TEXT\DATA\CLEANED\WOLOF_AUDIO_TRANS\alffa\alffa_clean_df.csv', 
        sr = 16000, 
        preload_data = False, 
        val_size = None, 
        transform = None)
    
    vocab_dict = ds.get_vocab_dict()
    for k, v in vocab_dict.items():
        print(f'{k} - {v}')