import pandas as pd
import numpy as np
import soundfile as sf
import os

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, files, sep, sr, audio_column_name, duration_column_name, min_duration, max_duration):
        self.sep = sep
        self.sr = sr
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_column_name = audio_column_name
        self.duration_column_name = duration_column_name
        self.data = self.load_ds(files)
    
    def load_ds(self, all_files):
        li = []
        for filename in all_files.split(";"):
            df = pd.read_csv(filename, sep=self.sep, engine="python")
            li.append(df)
        data = pd.concat(li, axis=0, ignore_index=True)

        if self.duration_column_name in data.columns:
            data = data[data[self.duration_column_name] >= self.min_duration]
            print("Mean duration: ", data[self.duration_column_name].mean())
        return data

    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        batch = {}
        try:
            batch["input_values"] = sf.read(os.path.join(r"D:\MARONE\WOLOF\SPEECH_TO_TEXT", item[self.audio_column_name]))[0]
        except Exception as e:
            print(f"Error reading file {item[self.audio_column_name]}: {e}")
            return None
        

        if len(batch["input_values"])//self.sr > self.max_duration:
            start = np.random.randint(0, len(batch["input_values"]) - self.max_duration * self.sr)
            batch["input_values"] = batch["input_values"][start : start + int(self.max_duration * self.sr)]

        return batch