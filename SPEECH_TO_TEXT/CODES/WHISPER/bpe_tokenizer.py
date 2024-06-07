import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_bpe_tokenizer(corpus_files, save_directory):
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Setup a trainer with special tokens
    trainer = trainers.BpeTrainer()

    # Customize pre-tokenizer if needed, here we are using a simple one
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Train the tokenizer
    tokenizer.train(files=corpus_files, trainer=trainer)

    # Save the tokenizer files
    os.makedirs(save_directory, exist_ok=True)
    tokenizer.model.save(save_directory)

    # Save the tokenizer itself (optional, includes all parts)
    tokenizer.save(os.path.join(save_directory, "tokenizer.json"))

def main():
    corpus_files = [r"D:\MARONE\WOLOF\LM\NGRAM\DATA\CLEANED\data.txt"]
    save_directory = r"D:\MARONE\WOLOF\SPEECH_TO_TEXT\MODELS\WHISPER\tokenizer"

    # Train the BPE tokenizer
    train_bpe_tokenizer(corpus_files, save_directory)

if __name__ == "__main__":
    main()
