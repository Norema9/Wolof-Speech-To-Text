import argparse
import json
import torch.multiprocessing as mp
import sys
import os
import toml
import warnings
warnings.filterwarnings('ignore')
os.chdir(r"D:\MARONE\WOLOF\LM\PONCTUATION")
# to add the path of the different on module
sys.path.append('COD')

from utils.utils import *

def main(config):
    set_seed(config["meta"]["seed"])

    config["dataset_creator"]["args"]["special_tokens"] = config["special_tokens"]
    config["dataset_creator"]["args"]["word_sep"] = config["meta"]["word_sep"]
    config["dataset_creator"]["args"]["char_to_replace_dict"] = config["char_to_replace"]

    print(config["char_to_replace"])

    dataset_creator = initialize_module(config["dataset_creator"]["path"], args=config["dataset_creator"]["args"])
    dataset_creator.create()  # This is commented if the data is already generated
    vocab_dict = dataset_creator.get_vocab_dict()

    with open(config['meta']["vocab_file"], 'w+') as f:
        json.dump(vocab_dict, f)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='config file path (default: None)')     
    
    args = parser.parse_args()
    config = toml.load(args.config)
    
    main(config)

