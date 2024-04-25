import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import os
import toml
import warnings
import datetime
warnings.filterwarnings('ignore')
os.chdir(r"D:\MARONE\WOLOF\LM\PONCTUATION")
# to add the path of the different on module
sys.path.append('COD')

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from time import gmtime, strftime
from utils.utils import *
from utils.metric import Metric
from transformers import BertTokenizer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4444'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600 * 5))

def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, config, resume):
    os.environ['CUDA_VISIBLE_DEVICES'] = config["meta"]["device_ids"]
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    setup(rank, world_size)

    # pretrained_path = config["meta"]["pretrained_path"]
    epochs = config["meta"]["epochs"]
    gradient_accumulation_steps = config["meta"]["gradient_accumulation_steps"]
    use_amp = config["meta"]["use_amp"]
    max_clip_grad_norm = config["meta"]["max_clip_grad_norm"]
    save_dir =  os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/checkpoints')
    log_dir = os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/log_dir')
    
    if rank == 0:
        # Creatr dirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Store config file
        config_name = strftime("%Y-%m-%d %H~%M~%S", gmtime()).replace(' ', '_') + '.toml'
        with open(os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '\\' + config_name), 'w+', encoding = 'utf-8') as f:
            toml.dump(config, f)
            f.close()

    # This should be needed to be reproducible https://discuss.pytorch.org/t/setting-seed-in-torch-ddp/126638
    config["meta"]["seed"] += rank
    set_seed(config["meta"]["seed"])
    
    # Create tokenizer
    tokenizer = BertTokenizer(config["meta"]["vocab_file"], 
                                    **config["special_tokens"],
                                    word_delimiter_token = config["meta"]["word_sep"])

    config['train_dataset']['args']['rank'] = rank
    config['val_dataset']['args']['rank'] = rank

    config["train_dataset"]["args"]["dist"] = dist
    config["val_dataset"]["args"]["dist"] = dist

    config["train_dataset"]["args"]["text_file"] = config["dataset_creator"]['args']["save_data_dir"] + os.sep + config["dataset_creator"]['args']["train_text_file_name"]
    config["val_dataset"]["args"]["text_file"] = config["dataset_creator"]['args']["save_data_dir"] + os.sep + config["dataset_creator"]['args']["val_text_file_name"]

    config["train_dataset"]["args"]["labels_file"] = config["dataset_creator"]['args']["save_data_dir"] + os.sep + config["dataset_creator"]['args']["train_label_file_name"]
    config["val_dataset"]["args"]["labels_file"] = config["dataset_creator"]['args']["save_data_dir"] + os.sep + config["dataset_creator"]['args']["val_label_file_name"]

    config["train_dataset"]["args"]["tokenizer"] = tokenizer
    config["val_dataset"]["args"]["tokenizer"] = tokenizer

    config["train_dataset"]["args"]["max_seq_length"] = config["meta"]["max_seq_length"]
    config["val_dataset"]["args"]["max_seq_length"] = config["meta"]["max_seq_length"]

    config["train_dataset"]["args"]["verbose"] = config["meta"]["verbose"]
    config["val_dataset"]["args"]["verbose"] = config["meta"]["verbose"]
    
    config["train_dataset"]["args"]["capit_label_vocab_file"] = config["dataset_creator"]['args']["save_data_dir"] + os.sep + config["dataset_creator"]['args']["capit_label_vocab_file"]
    config["val_dataset"]["args"]["capit_label_vocab_file"] = config["dataset_creator"]['args']["save_data_dir"] + os.sep + config["dataset_creator"]['args']["capit_label_vocab_file"]

    config["train_dataset"]["args"]["pad_label"] = config["meta"]["pad_label"]
    config["val_dataset"]["args"]["pad_label"] = config["meta"]["pad_label"]

    config["train_dataset"]["args"]["batch_shuffling_random_seed"] = config["meta"]["seed"]
    config["val_dataset"]["args"]["batch_shuffling_random_seed"] = config["meta"]["seed"]

    config["train_dataset"]["args"]["label_info_save_dir"] = config["train_dataset"]['args']["label_info_save_dir"]
    config["val_dataset"]["args"]["label_info_save_dir"] = config["val_dataset"]['args']["label_info_save_dir"]


    config["model"]["args"]["pretrained_bert_model_name"] = config["meta"]["pretrained_bert_model_name"]
    config["model"]["args"]['num_punct'] = len(config["dataset_creator"]['args']["punctuations"]) + 1

    config["default_collate"]["args"]["tokenizer"] = tokenizer
    
    dist.barrier()

    train_base_ds = initialize_module(config["train_dataset"]["path"], args=config["train_dataset"]["args"])
    
  
    default_collate = initialize_module(config["default_collate"]["path"], args=config["default_collate"]["args"])

    # Create train dataloader
    train_ds= train_base_ds.get_data()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas = world_size,
        rank = rank,
        **config["train_dataset"]["sampler"]
    )
    train_dl = DataLoader(
        dataset = train_ds,
        **config["train_dataset"]["dataloader"],
        sampler = train_sampler,
        collate_fn = default_collate
    )

    # Create val dataloader
    val_base_ds = initialize_module(config["val_dataset"]["path"], args=config["val_dataset"]["args"])
    val_ds = val_base_ds.get_data()
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        **config["val_dataset"]["sampler"]
    )
    val_dl = DataLoader(
        dataset = val_ds,
        **config["val_dataset"]["dataloader"],
        sampler = val_sampler,
        collate_fn = default_collate
    )


    # Initialize model
    model = initialize_module(config["model"]["path"], args=config["model"]["args"])

    # DDP for multi-processing
    model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=True)

    # Set up metric, scheduler, optmizer
    compute_metric = Metric()
    optimizer = torch.optim.AdamW(
        params = model.parameters(),
        lr = config["optimizer"]["lr"]
    )
    steps_per_epoch = (len(train_dl)//gradient_accumulation_steps) + (len(train_dl)%gradient_accumulation_steps != 0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config["scheduler"]["max_lr"], 
        epochs=epochs, 
        steps_per_epoch = steps_per_epoch)


    if rank == 0:
        print("Number of training utterances: ", len(train_ds))
        print("Number of validation utterances: ", len(val_ds))

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)
    trainer = trainer_class(
        dist = dist,
        rank = rank,
        n_gpus = world_size,
        config = config,
        resume = resume,
        epochs = epochs,
        steps_per_epoch = steps_per_epoch,
        model = model,
        tokenizer = tokenizer,
        compute_metric = compute_metric,
        train_dl = train_dl,
        val_dl = val_dl,
        train_sampler = train_sampler,
        val_sampler = val_sampler,
        optimizer = optimizer,
        scheduler = scheduler,
        save_dir = save_dir,
        log_dir = log_dir,
        gradient_accumulation_steps = gradient_accumulation_steps,
        use_amp = use_amp,
        max_clip_grad_norm = max_clip_grad_norm
    )

    trainer.train()


    cleanup()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', action="store_true",
                      help='path to latest checkpoint (default: None)')        
    
    args = args.parse_args()
    config = toml.load(args.config)
    n_gpus = len(config['meta']["device_ids"].split(','))
    
    mp.spawn(
        main,
        args = (n_gpus, config, args.resume),
        nprocs = n_gpus,
        join = True
    )

