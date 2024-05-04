import argparse
import math
import os
import sys
from pathlib import Path
from tqdm import tqdm
import toml
import torch.multiprocessing as mp
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

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: float = 0.5

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

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)
        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask")
        )
        
        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch



def main(config, resume):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4444'
    #os.environ["CUDA_VISIBLE_DEVICES"] = config["meta"]["device_ids"]

    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    seed = config["meta"]["seed"]
    push_to_hub = config["huggingface"]["push_to_hub"]
    hub_model_id = config["huggingface"]["hub_model_id"]
    hub_token = config["huggingface"]["args"]["hub_token"]
    logging_steps = config["meta"]["logging_steps"]
    max_gumbel_temperature = config["meta"]["max_gumbel_temperature"]
    gumbel_temperature_decay = config["meta"]["gumbel_temperature_decay"]
    min_gumbel_temperature = config["meta"]["min_gumbel_temperature"]    
    saving_steps = config["meta"]["saving_steps"]
    pad_to_multiple_of = None
    max_train_steps = config["meta"]["max_train_steps"]
    num_train_epochs = config["meta"]["num_train_epochs"]
    per_device_train_batch_size = config["train_dataset"]["dataloader"]["per_device_train_batch_size"]
    gradient_accumulation_steps = config["meta"]["gradient_accumulation_steps"]
    mask_time_prob = config["meta"]["mask_time_prob"]
    
    
    output_dir =  os.path.join(config["meta"]["output_dir"], 'checkpoints')
    log_dir = os.path.join(config["meta"]["output_dir"], 'log_dir')
    config_dir = os.path.join(config["meta"]["output_dir"], 'configs')
    accelerator_dir = os.path.join(config["meta"]["output_dir"], 'acelerator')

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(dispatch_batches=False)
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # set up tensorboard if available
        writer = SummaryWriter(log_dir, max_queue=5, flush_secs=30)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if push_to_hub:
            if hub_model_id is None:
                repo_name = get_full_repo_name(Path(output_dir).name, token=hub_token)
            else:
                repo_name = hub_model_id
            repo = Repository(output_dir, clone_from=repo_name)
        elif output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(accelerator_dir, exist_ok=True)
            
        # Store config file
        config_name = strftime("%Y-%m-%d %H~%M~%S", gmtime()).replace(' ', '_') + '.toml'
        with open(os.path.join(config_dir, config_name), 'w+') as f:
            toml.dump(config, f)
            f.close()

                
    accelerator.wait_for_everyone()
    
    
    # # Download data
    # train_dataset = initialize_module(config["train_dataset"]["path"], args=config["train_dataset"]["args"])
    # val_dataset = initialize_module(config["val_dataset"]["path"], args=config["val_dataset"]["args"])

    train_dataset = CustomDataset(
        config["train_dataset"]["args"]["files"], 
        sep=config["train_dataset"]["args"]["sep"], 
        audio_column_name=config["train_dataset"]["args"]["audio_column_name"], 
        duration_column_name=config["train_dataset"]["args"]["duration_column_name"],
        sr=16000, 
        min_duration=config["train_dataset"]["args"]["min_duration"], 
        max_duration=config["train_dataset"]["args"]["max_duration"])

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
        model=model, feature_extractor=feature_extractor, pad_to_multiple_of = pad_to_multiple_of, mask_time_prob = mask_time_prob
    )

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=config["train_dataset"]["dataloader"]["per_device_train_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        # prefetch_factor=16
    )

    eval_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator, 
        batch_size=config["val_dataset"]["dataloader"]["per_device_eval_batch_size"],
        num_workers=16
    )

    # Optimizer
    optimizer =  torch.optim.AdamW(
        list(model.parameters()),
        lr=config["optimizer"]["learning_rate"],
        betas=[config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]],
        eps=config["optimizer"]["adam_epsilon"],
    )

    lr_scheduler = get_scheduler(
        name=config["scheduler"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=config["scheduler"]["num_warmup_steps"],
        num_training_steps=max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader 
    )
    if resume:
        print("******Resume checkpoint******")
        accelerator.load_state(accelerator_dir)
        checkpoint = torch.load(os.path.join(output_dir, 'latest_checkpoint.pt'), 
                                map_location="cpu")

    
    # Train
    total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)


    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch


    # Afterwards we recalculate our number of training epochs
    config["meta"]["num_train_epochs"] = math.ceil(max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        print("Number of training data: ", len(train_dataset))
        print("total_batch_size: ", total_batch_size)
        print("num_update_steps_per_epoch: ", num_update_steps_per_epoch)
        print("num_train_epochs: ", num_train_epochs)

    # Only show the progress bar once on each machine.
    completed_steps = checkpoint['completed_steps'] + 1 if resume else 0
    resume_steps = checkpoint['completed_steps'] + 1 if resume else 0
    starting_epoch = checkpoint['epoch'] if resume else 0
    progress_bar = tqdm(initial = completed_steps, total = max_train_steps, disable=not accelerator.is_local_main_process)

    print(f"******STARTING AT EPOCH {starting_epoch} - STEP {completed_steps}******")


    for epoch in range(starting_epoch, num_train_epochs):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}: ")
        model.train()
        
        if resume:
            train_dataloader = islice(train_dataloader, resume_steps, None) # start the train_loader from the last iteration
        
        for step, batch in enumerate(train_dataloader):
            num_losses = batch["mask_time_indices"].sum()
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()

            
            # forward
            outputs = model(**batch)
            
            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = outputs.loss / gradient_accumulation_steps
            accelerator.backward(loss)

            # make sure that `num_losses` is summed for distributed training
            # and average gradients over losses of all devices
            if accelerator.state.num_processes > 1:
                num_losses = accelerator.gather(num_losses).sum()
                gradient_multiplier = accelerator.state.num_processes / num_losses
                multiply_grads(model.module.parameters(), gradient_multiplier)
            else:
                multiply_grads(model.parameters(), 1 / num_losses)

            # update step
            if (step + 1) % gradient_accumulation_steps == 0:

                # compute grad norm for monitoring
                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1
                )
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.parameters(), scale)

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )

                # update gumbel temperature
                gumbel_temperature = max(
                    max_gumbel_temperature * gumbel_temperature_decay**completed_steps,
                    min_gumbel_temperature,
                )
                if hasattr(model, "module"):
                    model.module.set_gumbel_temperature(gumbel_temperature)
                else:
                    model.set_gumbel_temperature(gumbel_temperature)

                progress_bar.update(1)
                completed_steps += 1

            # Log all results
            if (step + 1) % (gradient_accumulation_steps * logging_steps) == 0:
                loss.detach()
                outputs.contrastive_loss.detach()
                outputs.diversity_loss.detach()
                cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)
                cosine_sim = cosine_sim[batch["mask_time_indices"].to(torch.bool)].mean()

                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather(loss).sum()
                    outputs.contrastive_loss = accelerator.gather(outputs.contrastive_loss).sum()
                    outputs.diversity_loss = accelerator.gather(outputs.diversity_loss).sum()
                    percent_masked = accelerator.gather(percent_masked).sum()
                    cosine_sim = accelerator.gather(cosine_sim).mean()

                train_logs = {
                    "step": torch.tensor((step + 1) // gradient_accumulation_steps, dtype=torch.int32),
                    "loss": (loss * gradient_accumulation_steps) / num_losses,
                    "contrast_loss": outputs.contrastive_loss / num_losses,
                    "div_loss": outputs.diversity_loss / num_losses,
                    "%_mask_idx": percent_masked / accelerator.num_processes,
                    "ppl": outputs.codevector_perplexity,
                    "lr": torch.tensor(lr_scheduler.get_lr()),
                    "temp": torch.tensor(gumbel_temperature),
                    "grad_norm": torch.tensor(grad_norm),
                    "cosine_sim": cosine_sim * 100
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)
                    for k, v in train_logs.items():
                        writer.add_scalar('TRAIN' + '/' + k, v, completed_steps)
                    

            # save model every `args.saving_steps` steps
            if (step + 1) % (gradient_accumulation_steps * saving_steps) == 0:
                if (push_to_hub and epoch < num_train_epochs - 1) or output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                            output_dir + f'/saved_model_epoch_{epoch}', is_main_process=accelerator.is_main_process, save_function=accelerator.save
                        )
                    if accelerator.is_main_process:
                        feature_extractor.save_pretrained(output_dir + f'/saved_model_epoch_{epoch}')
                        print("****Saving checkpoint*****")
                        state_dict = {
                            "completed_steps": completed_steps,
                            "epoch": epoch
                        }
                        torch.save(state_dict, os.path.join(output_dir, "latest_checkpoint.pt"))
                    accelerator.save_state(accelerator_dir)

                if (push_to_hub and epoch < num_train_epochs - 1) and accelerator.is_main_process:
                    repo.push_to_hub(
                        commit_message=f"Training in progress step {completed_steps}",
                        blocking=False,
                        auto_lfs_prune=True,
                    )

            # if completed steps > `args.max_train_steps` stop
            if completed_steps >= max_train_steps:
                break

        print("******END OF EPOCH******")
        # Validate!
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
                writer.add_scalar('VALIDATION' + '/' + k, v, epoch)

            

        if output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                    output_dir + f'/saved_model_epoch_{epoch}', is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
            if accelerator.is_main_process:
                feature_extractor.save_pretrained(output_dir + f'/saved_model_epoch_{epoch}')
                print("****Saving checkpoint*****")
                state_dict = {
                    "completed_steps": completed_steps,
                    "epoch": epoch
                }
                torch.save(state_dict, os.path.join(output_dir, "latest_checkpoint.pt"))

            accelerator.save_state(accelerator_dir)
            if accelerator.is_main_process:
                if push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', action="store_true",
                      help='path to latest checkpoint (default: None)')         
    
    args = args.parse_args()
    config = toml.load(args.config)
    
    main(config, args.resume)