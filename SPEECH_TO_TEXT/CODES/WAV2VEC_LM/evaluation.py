from ctypes import Union
from typing import Any
import torch

from tqdm import tqdm
from torch.cuda.amp import autocast
import datetime
from typing import Dict, Union
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from torch.utils.data import DataLoader
from utils.metric import Metric
from utils.utils import *
from dataloader.dataset import DefaultCollate
import torch.distributed as dist
from time import gmtime, strftime
import json
import argparse
import torch.multiprocessing as mp
import sys
import os
import toml
import warnings

warnings.filterwarnings('ignore')
os.chdir(r"D:\MARONE\WOLOF\SPEECH_TO_TEXT")
# to add the path of the different on module
sys.path.append(r'CODES\WAV2VEC_LM')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4444'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600 * 5))

def cleanup():
    dist.destroy_process_group()

def gather(value: torch.tensor) -> Any:
        # gather value across devices - https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather
        if value.ndim == 0:
            value = value.clone()[None]
        output_tensors = [value.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(output_tensors, value)
        return torch.cat(output_tensors, dim=0)
    
def preload_model(rank, model_path, model) -> None:
    """
    Preload model parameters (in "*.tar" format) at the start of experiment.
    Args:
        model_path: The file path of the *.tar file
    """
    assert os.path.exists(model_path), f"The file {model_path} is not exist. please check path."
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(model_path, map_location=map_location)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint["model"], strict = False)
    else:
        model.load_state_dict(checkpoint["model"], strict = False)
    print(f"Model preloaded successfully from {model_path}.")
    
    if rank == 0:
        print(f"Model preloaded successfully from {model_path}.")
    
    return model
    
def eval(rank, val_dl, model, use_amp, compute_metric, n_gpus) -> Dict[str, Union[Any, float]]:
    # init logs
    val_logs = {
        "loss": 0,
        "wer": 0
    }

    model.eval()
    
    for batch in tqdm(val_dl, total = len(val_dl), disable = not rank == 0):
        with torch.no_grad():
            with autocast(enabled = use_amp):
                outputs = model(**batch)

        val_logs["loss"] += outputs.loss / len(val_dl)
        val_logs["wer"] += torch.tensor(compute_metric(outputs.logits, batch['labels'])) / len(val_dl)

    # average over devices in ddp
    if n_gpus > 1:
        val_logs = {k: gather(v).mean() for k, v in val_logs.items()}
    val_logs = {k: v.item() if hasattr(v, 'item') else v for k, v in val_logs.items()}
    return val_logs

def main(rank, world_size, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config["meta"]["device_ids"]
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    setup(rank, world_size)
    
    
    use_amp = config["meta"]["use_amp"]
    log_dir = os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/log_dir')
    huggingface_folder = config["meta"]["huggingface_folder"]
    model_path = config["meta"]["huggingface_hub_model"]
    best_model_path = config["meta"]["best_model"]
    
    
    
    if rank == 0:
        # Creatr dirs
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Store config file
        config_name = strftime("%Y-%m-%d %H~%M~%S", gmtime()).replace(' ', '_') + '.toml'
        with open(os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '\\' + config_name), 'w+') as f:
            toml.dump(config, f)
            f.close()
            
    
    config["meta"]["seed"] += rank
    set_seed(config["meta"]["seed"])
    config['val_dataset']['args']['sr'] = config['meta']['sr']
    config['val_dataset']['args']['rank'] = rank
    config["val_dataset"]["args"]["dist"] = dist
    config["val_dataset"]["args"]["special_tokens"] = config["special_tokens"]
    
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(huggingface_folder)
    model = Wav2Vec2ForCTC.from_pretrained(best_model_path)
    #model = preload_model(rank, best_model_path, model)
    compute_metric = Metric(processor)
    
    # Create val dataloader
    val_base_ds = initialize_module(config["val_dataset"]["path"], args=config["val_dataset"]["args"])
    val_ds = val_base_ds.get_data()
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        **config["val_dataset"]["sampler"]
    )
    
    default_collate = DefaultCollate(processor, config['meta']['sr'])
    val_dl = DataLoader(
        dataset=val_ds,
        **config["val_dataset"]["dataloader"],
        sampler = val_sampler,
        collate_fn=default_collate
    )
    
    eval_result = eval(rank, val_dl, model, use_amp, compute_metric, world_size)

    with open(os.path.join(config["meta"]["save_dir"], config["meta"]["name"], "evaluation_result.json"), 'w') as fp:
        json.dump(eval_result, fp)
    
    print(eval_result)
    
    cleanup()
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='config file path (default: None)')          
    
    args = args.parse_args()
    config = toml.load(args.config)
    n_gpus = len(config['meta']["device_ids"].split(','))
    
    mp.spawn(
        main,
        args = (n_gpus, config),
        nprocs = n_gpus,
        join = True
    )