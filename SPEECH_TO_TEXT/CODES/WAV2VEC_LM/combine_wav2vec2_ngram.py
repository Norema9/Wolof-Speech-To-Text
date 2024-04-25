from transformers import AutoProcessor
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2Processor
import argparse
import os
import toml
import sys
# from importlib import reload
# sys.setdefaultencoding() does not exist, here!
# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('UTF8')

os.chdir(r"D:\MARONE\WOLOF\SPEECH_TO_TEXT")

def build_lm_processor(ngram_path: str, processor_location:str, processor_with_lm_savedir:str):
    processor = Wav2Vec2Processor.from_pretrained(processor_location)
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path = ngram_path,
    )
    
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )
    
    processor_with_lm.save_pretrained(processor_with_lm_savedir)
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR INFERENCE ARGS')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='config file path (default: None)') 
    args = args.parse_args()
    
    config = toml.load(args.config)
    ngram_path = config['build_lm_processor']["ngram_path"]
    processor_location = config['build_lm_processor']["processor_location"]
    processor_with_lm_savedir = config['build_lm_processor']["processor_with_lm_savedir"]
    
    
    build_lm_processor(ngram_path, processor_location, processor_with_lm_savedir)
    
    

    
    
    
