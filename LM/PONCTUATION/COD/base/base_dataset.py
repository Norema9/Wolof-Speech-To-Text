import os
from pandarallel import pandarallel
from torch.utils.data import Dataset
from typing import Union, Optional
import multiprocessing as mp
from dataloader.dataset import Dataset as InstanceDataset
from transformers import BertTokenizer

class BaseDataset(Dataset):
    def __init__(self, 
                rank, 
                dist, 
                text_file: Union[str, os.PathLike],
                labels_file: Union[str, os.PathLike],
                tokenizer: BertTokenizer,
                max_seq_length: int,
                n_jobs: int,
                num_samples: int = -1,
                verbose: bool = True,
                punct_label_vocab_file: Optional[Union[str, os.PathLike]] = None,
                capit_label_vocab_file: Optional[Union[str, os.PathLike]] = None,
                pad_label: str = 'P',
                ignore_extra_tokens: bool = False,
                ignore_start_end: bool = True,
                tokenization_progress_queue: Optional[mp.Queue] = None,
                number_of_batches_is_multiple_of: int = 1,
                batch_shuffling_random_seed: int = 42,
                get_label_frequencies: bool = False,
                label_info_save_dir: Optional[Union[str, os.PathLike]] = None,
                nb_workers = 4):
        self.rank = rank
        self.dist = dist
        # Special characters to remove in your data 
        pandarallel.initialize(progress_bar = True, nb_workers = nb_workers)

        self.text_file = text_file
        self.labels_file = labels_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.n_jobs = n_jobs
        self.num_samples = num_samples
        self.verbose = verbose
        self.punct_label_vocab_file = punct_label_vocab_file
        self.capit_label_vocab_file = capit_label_vocab_file
        self.pad_label = pad_label
        self.ignore_extra_tokens = ignore_extra_tokens
        self.ignore_start_end = ignore_start_end
        self.tokenization_progress_queue = tokenization_progress_queue
        self.number_of_batches_is_multiple_of = number_of_batches_is_multiple_of
        self.batch_shuffling_random_seed = batch_shuffling_random_seed
        self.get_label_frequencies = get_label_frequencies
        self.label_info_save_dir = label_info_save_dir



    def get_data(self) -> Dataset:
        ds = InstanceDataset(self.text_file,
                            self.labels_file,
                            self.tokenizer,
                            self.max_seq_length,
                            self.n_jobs,
                            num_samples = self.num_samples,
                            verbose = self.verbose,
                            punct_label_vocab_file = self.punct_label_vocab_file,
                            capit_label_vocab_file = self.capit_label_vocab_file,
                            pad_label = self.pad_label,
                            ignore_extra_tokens = self.ignore_extra_tokens,
                            ignore_start_end = self.ignore_start_end,
                            tokenization_progress_queue = self.tokenization_progress_queue,
                            number_of_batches_is_multiple_of = self.number_of_batches_is_multiple_of,
                            batch_shuffling_random_seed = self.batch_shuffling_random_seed,
                            get_label_frequencies = self.get_label_frequencies,
                            label_info_save_dir = self.label_info_save_dir
                            )
        return ds
