import torch
from typing import Any, Dict, List, Union
from utils.feature import load_wav
from typing import Dict
from transformers import WhisperProcessor

class DefaultCollate:
    """ DataCollatorSpeechSeq2SeqWithPadding """
    def __init__(self, processor:WhisperProcessor, sr) -> None:
        self.processor = processor
        self.sr = sr

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class Dataset:
    def __init__(self, data, sr, preload_data, feature_extractor, transform = None):
        self.data = data
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data
        self.feature_extractor = feature_extractor
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx) -> tuple:
        item = self.data.iloc[idx]
        if not self.preload_data:
            audio = load_wav(item['path'], sr = self.sr)
        else:
            audio = item['wav']

        feature = self.feature_extractor(audio, sampling_rate = self.sr).input_features[0]

        features = {"input_features": feature, "labels": item["transcript"]}
        
        return features

