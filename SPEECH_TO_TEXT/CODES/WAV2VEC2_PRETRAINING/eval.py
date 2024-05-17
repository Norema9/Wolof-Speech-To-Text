import argparse
import math
import os
import sys
from pathlib import Path
from tqdm import tqdm
import toml
#import torch.multiprocessing as mp
import pandas as pd
from itertools import islice

import torch
from torch.utils.data.dataloader import DataLoader
from datasets import DatasetDict, concatenate_datasets, load_dataset, IterableDatasetDict
from torch.utils.tensorboard import SummaryWriter
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import soundfile as sf
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    get_scheduler,
    set_seed,
)
from transformers.utils import get_full_repo_name
import warnings
import datetime
warnings.filterwarnings('ignore')
os.chdir(r"D:\MARONE\WOLOF\SPEECH_TO_TEXT")
# to add the path of the different on module
sys.path.append(r'D:\MARONE\WOLOF\SPEECH_TO_TEXT\CODES\WAV2VEC2_PRETRAINING')

from utils.utils import *
from time import gmtime, strftime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers.utils import get_full_repo_name
import json





logger = get_logger(__name__)


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
        batch["input_values"] = sf.read(item[self.audio_column_name])[0]
        

        if len(batch["input_values"])//self.sr > self.max_duration:
            start = np.random.randint(0, len(batch["input_values"]) - self.max_duration * self.sr)
            batch["input_values"] = batch["input_values"][start : start + int(self.max_duration * self.sr)]

        return batch


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    
    config:Wav2Vec2Config
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: float = 0.6
        
        
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths
    
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask")
        )
        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch

def main(config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4444'

    seed = config["meta"]["seed"]
    pad_to_multiple_of = None
    max_train_steps = config["meta"]["max_train_steps"]
    mask_time_prob = config["meta"]["mask_time_prob"]
    
    
    output_dir =  os.path.join(config["meta"]["output_dir"], 'checkpoints')
    log_dir = os.path.join(config["meta"]["output_dir"], 'log_dir')
    config_dir = os.path.join(config["meta"]["output_dir"], 'configs')
    

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(dispatch_batches=False, mixed_precision = "fp16")

    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # set up tensorboard if available
        writer = SummaryWriter(log_dir, max_queue=5, flush_secs=30)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
            
        # Store config file
        config_name = strftime("%Y-%m-%d %H~%M~%S", gmtime()).replace(' ', '_') + '.toml'
        with open(os.path.join(config_dir, config_name), 'w+') as f:
            toml.dump(config, f)
            f.close()

                
    accelerator.wait_for_everyone()

    val_dataset = CustomDataset(
        config["val_dataset"]["args"]["files"], 
        sep=config["val_dataset"]["args"]["sep"], 
        audio_column_name=config["val_dataset"]["args"]["audio_column_name"], 
        duration_column_name=config["val_dataset"]["args"]["duration_column_name"],
        sr=16000, 
        min_duration=config["val_dataset"]["args"]["min_duration"], 
        max_duration=config["val_dataset"]["args"]["max_duration"])


    # Load feature_extractor
    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["meta"]["model_name_or_path"])
    # only normalized-inputs-training is supported
    if not feature_extractor.do_normalize:
        raise ValueError(
            "Training is only supported for normalized inputs. Make sure ``feature_extractor.do_normalize == True``"
        )

    # Load model config
    # config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)
    config_wav = Wav2Vec2Config.from_pretrained(config["meta"]["model_name_or_path"])
    if not config_wav.do_stable_layer_norm or config_wav.feat_extract_norm != "layer":
        raise ValueError(
            "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and"
            " ``config.feat_extract_norm='layer'"
        )


    # initialize random model
    model = Wav2Vec2ForPreTraining(config_wav)
    if config["meta"]["load_from_pretrained"] is not None:
        try:
            model = model.from_pretrained(config["meta"]["model_name_or_path"])
        except:
            print("!!!!! Warning: Pretrained model may not exist. Start training from Scratch")
    
    # Activate gradient checkpointing if needed
    if config["meta"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # Define data collator, optimizer and scheduler
    data_collator = DataCollatorForWav2Vec2Pretraining(
        config=config_wav, feature_extractor=feature_extractor, pad_to_multiple_of = pad_to_multiple_of, mask_time_prob = mask_time_prob
    )

    eval_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator, 
        batch_size=config["val_dataset"]["dataloader"]["per_device_eval_batch_size"],
        num_workers=16
    )

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader 
    )

    progress_bar = tqdm(initial = 0, total = max_train_steps, disable=not accelerator.is_local_main_process)

    print(f"******STARTING AT EVALUATION ******")
    model.eval()

    # init logs
    val_logs = {
        "val_loss": 0,
        "val_contrastive_loss": 0,
        "val_diversity_loss": 0,
        "val_num_losses": 0,
    }
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch.pop("sub_attention_mask", None)
            outputs = model(**batch)

        val_logs["val_loss"] += outputs.loss
        val_logs["val_contrastive_loss"] += outputs.contrastive_loss
        val_logs["val_diversity_loss"] += outputs.diversity_loss
        val_logs["val_num_losses"] += batch["mask_time_indices"].sum()

    # sum over devices in multi-processing
    if accelerator.num_processes > 1:
        val_logs = {k: accelerator.gather(v).sum() for k, v in val_logs.items()}

    val_logs = {k: v / val_logs["val_num_losses"] for k, v in val_logs.items()}

    log_str = ""
    for k, v in val_logs.items():
        log_str += "| {}: {:.3e}".format(k, v.item())

    if accelerator.is_local_main_process:
        progress_bar.write(log_str)
        for k, v in val_logs.items():
            writer.add_scalar('VALIDATION' + '/' + k, v)

        

    if output_dir is not None:
        accelerator.wait_for_everyone()
        with open(os.path.join(output_dir, "evaluation_result.json"), 'w') as fp:
            json.dump(val_logs, fp)
    
        print(val_logs)
        
        
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='config file path (default: None)')       
    
    args = args.parse_args()
    config = toml.load(args.config)
    
    main(config)