import sys
import torch
import numpy as np
import pandas as pd

# from utils.feature import load_wav
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from pathlib import Path
import multiprocessing as mp
from time import sleep
import logging
from tqdm import tqdm
from queue import Empty
import itertools
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import os
import sys


MAX_NUM_QUERIES_IN_SPLIT = 10 ** 4
TOKENIZATION_PROGRESS_REPORT_PERIOD = 10 ** 3
BATCH_MARK_UP_PROGRESS_REPORT_PERIOD = 10 ** 4
BATCH_BUILDING_PROGRESS_REPORT_PERIOD = 10 ** 4


class Progress:
    """
    Manages several ``tqdm`` progress bars for multiprocess tasks. This class can be used as context manager.

    The class starts separate process which creates and updates progress bars. Information to progress process is
    passed via multiprocessing queues. There is a separate queue for every progress bar.

    You can use it as context manager:

    .. code-block:: python
        with Progress([10, 20], ["progress bar 1", "progress bar 2"], ["parrot", "frog"]) as progress_queues:
            num_processes = 10
            with multiprocessing.Pool(num_processes) as pool:
                data = list(zip(my_data, [progress_queues[0]] * num_processes, [progress_queues[1]] * num_processes))
                pool.starmap(worker_func, data)

    Or without context manager:

    .. code-block:: python
        progress = Progress([10, 20], ["progress bar 1", "progress bar 2"], ["parrot", "frog"])
        progress_queues = progress.get_queue()
        num_processes = 10
        with multiprocessing.Pool(num_processes) as pool:
            data = list(zip(my_data, [progress_queues[0]] * num_processes, [progress_queues[1]] * num_processes))
            pool.starmap(worker_func, data)
        progress.finish()

    In a worker function you will have to put number of processed items into the progress queues. For example:

    .. code-block:: python
        def worker_func(my_datum, parrot_progress_queue, frog_progress_queue):
            ...
            for i in range(10):
                parrot_progress_queue.put(1)
                frog_progress_queue.put(2)

    Progress bars and progress process are closed when ``finish`` or ``__exit__`` methods are called.
    """

    def __init__(self, total: Union[int, List[int]], desc: Union[str, List[str]], unit: Union[str, List[str]]) -> None:
        """
        Starts progress process and creates queues for passing information to the progress process. Number of progress
        bars is equal to the max length of lists ``total``, ``desc``, ``unit``. If none of these parameters is a list,
        then 1 progress bar is created.

        Args:
            total: a list of ``int`` which length is equal to the number of progress bars OR an ``int`` OR a list of
                one ``int``. Number which comprises 100% of progress bar. When sum of values passed through the
                corresponding queue equals ``total`` corresponding progress bar reaches 100%. If ``total`` is an
                ``int`` or a list of one element, then all progress bars have equal ``total`` parameter.
            desc: a list of ``str`` which length is equal to the number of progress bars OR a ``str`` OR a list of one
                ``str``. Description of a progress bar which is showed as a prefix. See more in description of
                parameter ``desc`` of function ``tqdm.tqdm``.
            unit: a list of ``str`` which length is equal to the number of progress bars OR a ``str`` OR a list of one
                ``str``. A unit of a progress bar. See more in description of parameter ``unit`` of function
                ``tqdm.tqdm``.
        """
        if not isinstance(total, list):
            total = [total]
        if not isinstance(desc, list):
            desc = [desc]
        if not isinstance(unit, list):
            unit = [unit]
        num_processes = max([len(total), len(desc), len(unit)])
        for param in [total, desc, unit]:
            if len(param) not in [num_processes, 1]:
                raise ValueError(
                    f"If parameter of `Progress.__init__` method is a list, then it has to be the same length as other "
                    f"parameters which are lists"
                )
            if len(param) == 1:
                param *= num_processes
        manager = mp.Manager()
        self.progress_queues = tuple(manager.Queue() for _ in range(num_processes))
        self.progress_process = mp.Process(target=_show_prog, args=(self.progress_queues, total, desc, unit))
        self.progress_process.start()


    def get_queues(self):
        return self.progress_queues

    def finish(self):
        """
        Finish the progress bars and close the progress process.
        """
        # Terminate the progress process
        self.progress_process.terminate()

        # # Join the progress process
        self.progress_process.join()

        # # Join the progress queues' threads
        # for queue in self.progress_queues:
        #     queue.join()

class TokenizeCreateMasksClipWorker:
    """A worker for tokenization, encoding labels, creating masks for first token in a word, sequence clipping"""

    def __init__(
        self,
        max_seq_length: int,
        tokenizer: BertTokenizer,
        punct_label_ids: Optional[Dict[str, int]],
        capit_label_ids: Optional[Dict[str, int]],
        pad_label: str,
        verbose: bool,
        progress_queue: mp.Queue,
    ) -> None:
        """
        Args:
            max_seq_length: max number of tokens in an input sequence including [CLS] and [SEP] tokens. If number of
                tokens in a sequence exceeds ``max_seq_length``, then excess tokens in the end of the sequence
                are removed
            tokenizer: a tokenizer instance which has properties ``cls_id``, ``pad_id``, ``sep_id``, ``unk_id``
            punct_label_ids: dict to map punctuation labels to label ids. Starts with pad_label->0.
            capit_label_ids: dict to map capitalization labels to label ids. Starts with pad_label->0.
            pad_label: pad value use for labels. By default, it's the neutral label for punctuation and capitalization.
                Its id in ``punct_label_ids`` and ``capit_label_ids`` has to be ``0``
            verbose: whether to report when the worker finishes its job
            progress_queue: a multiprocessing queue used for reporting progress. Useful for creating tarred dataset
        """
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.punct_label_ids = punct_label_ids
        self.capit_label_ids = capit_label_ids
        self.pad_label = pad_label
        self.verbose = verbose
        self.progress_queue = progress_queue

    def _maybe_clip(self, values: List[int], append_value: int) -> List[int]:
        if len(values) > self.max_seq_length:
            return values[: self.max_seq_length - 1] + [append_value]
        return values

    def __call__(
        self,
        queries: List[str],
        punct_label_lines: Optional[Union[List[str], Tuple[str, ...]]],
        capit_label_lines: Optional[Union[List[str], Tuple[str, ...]]],
        split_i: int,
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        """
        Tokenize, clip, encode labels, and create masks of first tokens in words.

        Args:
            queries: text sequences
            punct_label_lines: a list or a tuple of labels for every word in a sequence (str)
            capit_label_lines: a list of a tuple labels for every word in a sequence (str)
            split_i: number of a split which is processed. Used for logging
            audio_queries: a list of audio filepaths
            sample_rate: target sample rate of audios
            preload_audios: whether to preload audios or not

        Returns:
            input_ids: a list of 1D int32 arrays. Each array contains token ids of the corresponding query
            subtokens_mask: a list of 1D boolean arrays. An array element is ``True`` if corresponding token is the
                first token in a word
            punct_labels: a list of 1D int32 arrays. Encoded punctuation labels for every token in a query. Tokens in
                one word have identical labels
            capit_labels: a list of 1D int32 arrays. Encoded capitalization labels for every token in a query. Tokens
                in one word have identical labels
        """
        all_input_ids, all_subtokens_mask, punct_all_labels, capit_all_labels = [], [], [], []
        dummy = [None] * len(queries)  # Needed to avoid code duplication with different values of `self.use_audio`
        progress_made = 0
        queries = zip(queries, dummy)
        for i, (query, _) in enumerate(queries):
            words = query.split()
            input_ids, subtokens_mask = [self.tokenizer.cls_token_id], [0]
            _check_number_of_labels(words, query, i, split_i, punct_label_lines[i], capit_label_lines[i])
            pad_id = self.punct_label_ids[self.pad_label]
            punct_labels = [pad_id]
            punct_query_labels = [self.punct_label_ids[lab] for lab in punct_label_lines[i]]
            capit_labels = [pad_id]
            capit_query_labels = [self.capit_label_ids[lab] for lab in capit_label_lines[i]]
            for j, word in enumerate(words):
                word_ids = self.tokenizer.convert_tokens_to_ids(word)
                if not word_ids and len(word):
                    word_ids = [self.tokenizer.unk_token_id]
                input_ids.extend(word_ids)

                subtokens_mask.append(1)
                subtokens_mask.extend([0] * (len(word_ids) - 1))

                punct_labels.extend([punct_query_labels[j]] * len(word_ids))
                capit_labels.extend([capit_query_labels[j]] * len(word_ids))

            # add eos token
            input_ids.append(self.tokenizer.sep_token_id)
            subtokens_mask.append(0)

            all_input_ids.append(np.array(self._maybe_clip(input_ids, self.tokenizer.sep_token_id), dtype=np.int32))
            all_subtokens_mask.append(np.array(self._maybe_clip(subtokens_mask, 0), dtype=bool))

            punct_labels.append(pad_id)
            punct_all_labels.append(np.array(self._maybe_clip(punct_labels, pad_id), dtype=np.int32))
            capit_labels.append(pad_id)
            capit_all_labels.append(np.array(self._maybe_clip(capit_labels, pad_id), dtype=np.int32))

            progress_made += 1
            if progress_made >= TOKENIZATION_PROGRESS_REPORT_PERIOD:
                self.progress_queue.put(progress_made)
                progress_made = 0

        self.progress_queue.put(progress_made)
        if self.verbose:
            logging.info(f"Finished processing data split number {split_i}")

        return (
            all_input_ids,
            all_subtokens_mask,
            punct_all_labels,
            capit_all_labels,
        )



def _show_prog(queues: Tuple[mp.Queue, ...], totals: List[int], descriptions: List[str], units: List[str]) -> None:
    """
    Show several ``tqdm`` progress bars.
    Args:
        queues: a list of queues by which progress is delivered into this function. Each queue is responsible for one
            progress bar. ``show_prog`` function extracts integers from ``queues`` elements and adds them to progress
            bars. If value extracted from a queue equals ``-1``, then corresponding progress bar is closed. When all
            progress bars are closed, this function returns.
        totals: list of values 100% of progress bars. See more in a description of ``total`` parameter of
            ``tqdm.tqdm`` function
        descriptions: list of descriptions of progress bars. See more in a description of ``desc`` parameter of
            ``tqdm.tqdm`` function
        units: list of progress bar units. See more in a description of ``unit`` parameter of ``tqdm.tqdm`` function
    """
    if not all([len(queues) == len(v) for v in [totals, descriptions, units]]):
        raise ValueError(
            f"All of parameters `queues`, `total_num_lines`, `descriptions`, `units` have to have equal lengths. "
            f"len(queues)={len(queues)}, len(total_num_lines)={len(totals)}, "
            f"len(descriptions)={len(descriptions)}, len(units)={len(units)}."
        )
    prog = [
        tqdm(total=tt, desc=dd, unit=uu, unit_scale=True, position=i)
        for i, (tt, dd, uu) in enumerate(zip(totals, descriptions, units))
    ]
    finished = [False] * len(queues)
    while True:
        for i, queue in enumerate(queues):
            stop = False
            to_add = 0
            try:
                v = queue.get(block=False)
                while v != -1:
                    to_add += v
                    v = queue.get(block=False)
                stop = True
            except Empty:
                if to_add == 0 and not stop:
                    continue
            prog[i].n += to_add
            prog[i].update(0)
            if prog[i].n >= totals[i]:
                finished[i] = True
                prog[i].close()
            if stop:
                if prog[i].n < totals[i]:
                    logging.warning(
                        f"Progress with description '{descriptions[i]}' terminated before progress bar "
                        f"reached 100%. prog.n={prog[i].n}, total_num_lines={totals[i]}"
                    )
                finished[i] = True
                prog[i].close()
        if all(finished):
            break
        sleep(0.1)

def create_masks_and_segment_ids(
    input_ids: np.ndarray,
    subtokens_mask: np.ndarray,
    pad_id: int,
    cls_id: int,
    sep_id: int,
    ignore_start_end: bool,
    ignore_extra_tokens: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates segment ids array, input mask, loss mask.

    Segment ids array is BERT token type ids in HuggingFace terminology. It is a zeros array for punctuation
    and capitalization task.

    Input mask element is ``True`` if an element of ``input_ids`` is not padding and ``False`` otherwise.

    Loss mask element is ``True`` for the first token in a word. If ``ignore_start_end=False``, then loss mask
    element is ``True`` for [CLS] and [SEP] tokens. If ``ignore_extra_tokens=False``, then loss mask element is ``True``
    for all word tokens. In all other cases loss mask elements are ``False``.

    Args:
        input_ids: an integer array of shape ``[Batch, Time]`` containing ids of source token ids
        subtokens_mask: a boolean array of shape ``[Batch, Time]`` which elements are ``True`` if they correspond to
            the first token of some word
        pad_id: an id of padding token
        cls_id: an id of [CLS] token
        sep_id: an id of [SEP] token
        ignore_start_end: whether to compute loss for [CLS] and [SEP] tokens
        ignore_extra_tokens: whether to compute loss for not first tokens in words

    Returns:
        segment_ids: int8 array of shape [Batch, Time]
        input_mask: boolean array of shape [Batch, Time]
        loss_mask: boolean array of shape [Batch, Time]
    """
    segment_ids = np.zeros_like(input_ids, dtype=np.int8)
    input_mask = np.not_equal(input_ids, pad_id)
    special_mask = np.equal(input_ids, cls_id) & np.equal(input_ids, sep_id)
    if ignore_start_end:
        if ignore_extra_tokens:
            loss_mask = subtokens_mask
        else:
            loss_mask = input_mask & ~special_mask
    else:
        if ignore_extra_tokens:
            loss_mask = subtokens_mask | special_mask
        else:
            loss_mask = input_mask
    return segment_ids, input_mask, loss_mask

def get_stats(lengths):
    total_samples = len(lengths)
    max_length = max(lengths)
    min_length = min(lengths)
    avg_length = sum(lengths) / total_samples
    logging.info(f"Total Samples: {total_samples}")
    logging.info(f"Maximum Length: {max_length}")
    logging.info(f"Minimum Length: {min_length}")
    logging.info(f"Average Length: {avg_length}")

def get_label_stats(labels, save_path):
    # Calculate label counts
    label_counts = {}
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    # Create DataFrame from label counts
    df = pd.DataFrame(list(label_counts.items()), columns=['Label', 'Count'])
    
    # Save DataFrame to TSV file
    df.to_csv(save_path, sep='\t', index=False)

def _get_features(
    queries: Union[List[str], Tuple[str, ...]],
    punct_label_lines: Union[List[str], Tuple[str, ...]],
    capit_label_lines: Union[List[str], Tuple[str, ...]],
    max_seq_length: int,
    tokenizer: BertTokenizer,
    punct_label_ids: Dict[str, int] = None,
    capit_label_ids: Dict[str, int] = None,
    pad_label: str = 'P',
    verbose: bool = True,
    n_jobs: Optional[int] = 0,
    progress_queue: Optional[mp.Queue] = None,
) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[Any], List[Any], List[Any]]:
    """
    Tokenizes data, encodes labels, creates masks of first tokens in words, clips sequences by number of tokens.

    Args:
        queries: text sequences
        max_seq_length: max number of tokens in an input sequence including [CLS] and [SEP] tokens. If number of tokens
            in a sequence exceeds ``max_seq_length``, then excess tokens in the end of the sequence are removed
        tokenizer: a tokenizer instance which has properties ``cls_id``, ``pad_id``, ``sep_id``, ``unk_id``
        punct_label_ids: dict to map punctuation labels to label ids. Starts with pad_label->0.
        capit_label_ids: dict to map capitalization labels to label ids. Starts with pad_label->0.
        pad_label: pad value use for labels. By default, it's the neutral label for punctuation and capitalization.
            Its id in ``punct_label_ids`` and ``capit_label_ids`` has to be ``0``
        punct_label_lines: a list of a tuple of labels for every word in a sequence (str)
        capit_label_lines: a list or a tuple of labels for every word in a sequence (str)
        verbose: whether to show examples of tokenized data and various progress information
        n_jobs: a number of workers used for preparing features. If ``n_jobs <= 0``, then do not use multiprocessing
            and run features creation in this process. If not set, number of workers will be equal to the number of
            CPUs.

            !!WARNING!!
            There can be deadlocking problems with some tokenizers (e.g. SentencePiece, HuggingFace AlBERT)
            if ``n_jobs > 0``.

        progress_queue: a multiprocessing queue used for reporting progress. Useful for creating tarred dataset
        audio_queries: a list of audio filepaths
        sample_rate: target sample rate of audios
        preload_audios: whether to preload audios or not

    Returns:
        input_ids: a list of 1D int32 arrays. Each array contains token ids of corresponding query
        subtokens_mask: a list of 1D boolean arrays. An array element is ``True`` if corresponding token is the
            first token in a word
        punct_labels: a list of 1D int32 arrays. Encoded punctuation labels for every token in a query. Tokens in one
            word have identical labels.
        capit_labels: a list of 1D int32 arrays. Encoded capitalization labels for every token in a query. Tokens in
            one word have identical labels
    """
    if verbose:
        logging.info("Start initial tokenization.")
    create_progress_process = progress_queue is None
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), len(queries))

    if verbose:
        logging.info(f"Running tokenization with {n_jobs} jobs.")

    # Number of queries in split
    split_size = min(len(queries) // max(n_jobs, 1), MAX_NUM_QUERIES_IN_SPLIT)
    n_split = len(queries) // split_size
    split_queries = [queries[split_size * i : split_size * (i + 1)] for i in range(n_split - 1)] + [
        queries[split_size * (n_split - 1) :]
    ]
    split_punct_labels_lines = [
        punct_label_lines[split_size * i : split_size * (i + 1)] for i in range(n_split - 1)
    ] + [punct_label_lines[split_size * (n_split - 1) :]]
    split_capit_labels_lines = [
        capit_label_lines[split_size * i : split_size * (i + 1)] for i in range(n_split - 1)
    ] + [capit_label_lines[split_size * (n_split - 1) :]]

    args = list(zip(split_queries, split_punct_labels_lines, split_capit_labels_lines, range(n_split)))
    
    if create_progress_process:
        progress = Progress(len(queries), "Tokenization", "query")
        progress_queue = progress.get_queues()[0]
    if n_jobs > 0:
        with mp.Pool(n_jobs) as pool:
            result = pool.starmap(
                TokenizeCreateMasksClipWorker(
                    max_seq_length, tokenizer, punct_label_ids, capit_label_ids, pad_label, verbose, progress_queue,
                ),
                args,
            )
    else:
        result = []
        for x in args:
            result.append(
                TokenizeCreateMasksClipWorker(
                    max_seq_length, tokenizer, punct_label_ids, capit_label_ids, pad_label, verbose, progress_queue,
                )(*x)
            )
    if create_progress_process:
        progress.finish()

    input_ids, subtokens_mask, punct_labels, capit_labels = tuple(
        list(itertools.chain(*e)) for e in zip(*result)
    )
    if verbose:
        logging.info("Finished initial tokenization.")
        get_stats([len(inp) for inp in input_ids])
        logging.info(f"Finished clipping and padding.")
        for i in range(min(len(input_ids), 5)):
            logging.info("*** Example ***")
            logging.info("i: %s" % i)
            logging.info("subtokens: %s" % " ".join(list(map(str, input_ids[i]))))
            logging.info("subtokens_mask: %s" % " ".join(list(map(str, subtokens_mask[i]))))
            logging.info("punct_labels: %s" % " ".join(list(map(str, punct_labels[i]))))
            logging.info("capit_labels: %s" % " ".join(list(map(str, capit_labels[i]))))

    return (
        input_ids,
        subtokens_mask,
        punct_labels,
        capit_labels,
    )

def load_label_ids(file_path: Union[str, os.PathLike]) -> Dict[str, int]:
    ids = {}
    with open(file_path, encoding='utf_8') as f:
        for i, line in enumerate(f):
            ids[line.strip()] = i
    return ids

def create_label_ids(unique_labels: Set[str], pad_label: str) -> Dict[str, int]:
    """
    Returns label ids dictionary. ``pad_label`` always has id ``0``. Other labels are sorted in alphabetical order.
    Args:
        unique_labels: a set of labels from which label ids dictionary is created. May or may not contain ``pad_label``
        pad_label: label used for padding. It is also a neutral label

    Returns:
        label ids dictionary
    """
    label_ids = {pad_label: -100}   ## The pad_label_id should be ignored during the loss computation
    if pad_label in unique_labels:
        unique_labels.remove(pad_label)
    for label in sorted(unique_labels):
        label_ids[label] = len(label_ids) - 1
    return label_ids

def _check_number_of_labels(
    words: List[str],
    query: str,
    qi: int,
    split_i: int,
    punctuation_labels: List[str],
    capitalization_labels: List[str],
) -> None:
    if len(words) != len(punctuation_labels):
        raise ValueError(
            f"Number of punctuation labels for a query number {qi} in a split number {split_i} is not equal to "
            f"number of words. Number of words: {len(words)}, number of punctuation labels: "
            f"{len(punctuation_labels)}. First 100 characters of the query: '{query[:100]}', punctuation labels: "
            f"'{punctuation_labels}'"
        )
    if len(words) != len(capitalization_labels):
        raise ValueError(
            f"Number of capitalization labels for a query number {qi} in a split number {split_i} is not equal to "
            f"number of words. Number of words: {len(words)}, number of capitalization labels: "
            f"{len(capitalization_labels)}. First 100 characters of the query: '{query[:100]}', "
            f"capitalization labels: '{capitalization_labels}'"
        )

class DefaultCollate:
    def __init__(self, tokenizer, ignore_start_end, ignore_extra_tokens) -> None:
        self.tokenizer: BertTokenizer = tokenizer
        self.ignore_start_end = ignore_start_end
        self.ignore_extra_tokens = ignore_extra_tokens


    def __call__(self, batches: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        for batch in batches:
            batch_segment_ids, batch_input_mask, batch_loss_mask = create_masks_and_segment_ids(
                batch['input_ids'],
                batch['subtokens_mask'],
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.ignore_start_end,
                self.ignore_extra_tokens,
            )
            batch['segment_ids'] = torch.as_tensor(batch_segment_ids, dtype=torch.int)
            batch['input_mask'] = torch.as_tensor(batch_input_mask)
            batch['loss_mask'] = torch.as_tensor(batch_loss_mask)
            batch['input_ids'] = torch.as_tensor(batch['input_ids'], dtype=torch.int)
            batch['subtokens_mask'] = torch.as_tensor(batch['subtokens_mask'])
            batch['punct_labels'] = torch.as_tensor(batch['punct_labels'], dtype=torch.long)
            batch['capit_labels'] = torch.as_tensor(batch['capit_labels'], dtype=torch.long)
            

        segment_ids = pad_sequence([batch['segment_ids'] for batch in batches])
        input_mask = pad_sequence([batch['input_mask'] for batch in batches])
        loss_mask = pad_sequence([batch['loss_mask'] for batch in batches])
        input_ids = pad_sequence([batch['input_ids'] for batch in batches], padding_value=self.tokenizer.pad_token_id)
        subtokens_mask = pad_sequence([batch['subtokens_mask'] for batch in batches], padding_value=False)
        punct_labels = pad_sequence([batch['punct_labels'] for batch in batches], padding_value=0)
        capit_labels = pad_sequence([batch['capit_labels'] for batch in batches], padding_value=0)
        return {
            'input_ids': input_ids.T,
            'subtokens_mask': subtokens_mask.T,
            'punct_labels': punct_labels.T,
            'capit_labels': capit_labels.T,
            'segment_ids': segment_ids.T,
            'input_mask': input_mask.T,
            'loss_mask': loss_mask.T,
        }

class Dataset:
    def __init__(self,
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
                label_info_save_dir: Optional[Union[str, os.PathLike]] = None
                ):
        self.text_file = text_file
        self.labels_file = labels_file
        self.tokenizer = tokenizer
        
        if punct_label_vocab_file is not None:
            punct_label_vocab_file = Path(punct_label_vocab_file).expanduser()
            punct_label_ids = load_label_ids(punct_label_vocab_file)
        else:
            punct_label_ids = {}
        if capit_label_vocab_file is not None:
            capit_label_vocab_file = Path(capit_label_vocab_file).expanduser()
            capit_label_ids = load_label_ids(capit_label_vocab_file)
        else:
            capit_label_ids = {}
        self.verbose = verbose
        self.pad_label = pad_label
        self.ignore_extra_tokens = ignore_extra_tokens
        self.ignore_start_end = ignore_start_end

        (
            text_lines,
            punct_label_lines,
            capit_label_lines,
            punct_unique_labels,
            capit_unique_labels,
        ) = self._read_dataset(self.text_file, self.labels_file, num_samples)  

        if punct_label_ids:
            self._check_label_ids_vs_unique_labels(
                punct_label_ids, punct_unique_labels, 'punct', 'punctuation', self.labels_file
            )
        else:
            punct_label_ids = create_label_ids(punct_unique_labels, self.pad_label)
        if capit_label_ids:
            self._check_label_ids_vs_unique_labels(
                capit_label_ids, capit_unique_labels, 'capit', 'capitalization', self.labels_file
            )
        else:
            capit_label_ids = create_label_ids(capit_unique_labels, self.pad_label)
        
        features = _get_features(
                text_lines,
                punct_label_lines,
                capit_label_lines,
                max_seq_length,
                self.tokenizer,
                pad_label = self.pad_label,
                punct_label_ids = punct_label_ids,
                capit_label_ids = capit_label_ids,
                verbose = self.verbose,
                progress_queue = tokenization_progress_queue,
                n_jobs = n_jobs,
            )
        #TODO : save the features in pikle files so data can be restored
        (
            self.input_ids,
            self.subtokens_mask,
            self.punct_labels,
            self.capit_labels,
        ) = features
        self.punct_label_ids, self.capit_label_ids = punct_label_ids, capit_label_ids
        self.number_of_batches_is_multiple_of = number_of_batches_is_multiple_of
        self.batch_shuffling_random_state = np.random.RandomState(batch_shuffling_random_seed)
        self.label_info_save_dir = label_info_save_dir
        if get_label_frequencies:
            self.punct_label_frequencies = self._calculate_and_save_label_frequencies(self.punct_labels, 'punct')
            self.capit_label_frequencies = self._calculate_and_save_label_frequencies(self.capit_labels, 'capit')
    
        self.batches = self._form_batches(
                input_ids = self.input_ids,
                subtokens_mask = self.subtokens_mask,
                punct_labels = self.punct_labels,
                capit_labels = self.capit_labels,
            )
    
    
    @staticmethod
    def _read_dataset(
        text_file: Path, 
        labels_file: Path, 
        num_samples: int
    ) -> Union[Tuple[Any, Any, Any, Set[Any], Set[Any], Any], Tuple[Any, Any, Any, Set[Any], Set[Any]]]:
        with open(text_file, 'r', encoding='utf_8') as f:
            text_lines = f.readlines()
        punct_unique_labels, capit_unique_labels = set(), set()
        punct_labels_lines, capit_labels_lines = [], []
        with open(labels_file, 'r', encoding='utf_8') as f:
            for i, line in enumerate(f):
                pairs = line.split()
                if not all([len(p) == 2 for p in pairs]):
                    raise ValueError(
                        f"Some label pairs are not pairs but have wrong length (!= 2) in line {i} in label file "
                        f"{labels_file}"
                    )
                words = text_lines[i].split()
                if len(pairs) != len(words):
                    raise ValueError(
                        f"In line {i} in text file {text_file} number of words {len(words)} is not equal to the "
                        f"number of labels {len(pairs)} in labels file {labels_file}."
                    )
                punct_line, capit_line = zip(*pairs)
                punct_labels_lines.append(punct_line)
                capit_labels_lines.append(capit_line)
                punct_unique_labels.update(punct_line)
                capit_unique_labels.update(capit_line)
        if len(punct_labels_lines) != len(text_lines):
            raise ValueError(
                f"Number of text lines {len(text_lines)} in text file {text_file} is not equal to the number of lines "
                f"{len(punct_labels_lines)} in labels file {labels_file}."
            )

        dataset = list(zip(text_lines, punct_labels_lines, capit_labels_lines))
            
        if len(dataset) == 0:
            raise ValueError(f"Dataset loaded from files {text_file} and {labels_file} is empty.")
        if num_samples > 0:
            dataset = dataset[:num_samples]
        text_lines, punct_labels_lines, capit_labels_lines = zip(*dataset)
        return text_lines, punct_labels_lines, capit_labels_lines, punct_unique_labels, capit_unique_labels
    


    @staticmethod
    def _check_label_ids_vs_unique_labels(
        label_ids: Dict[str, int], unique_labels: Set[str], label_type: str, task: str, label_file: Path
    ) -> None:
        if unique_labels - set(label_ids):
            not_present_labels = list(unique_labels - set(label_ids))
            raise ValueError(
                f"{len(not_present_labels)} {task} labels found in {label_file} are not present in "
                f"`{label_type}_label_ids`. Examples of unexpected labels from {label_file}: {not_present_labels[:3]}"
            )

    def _calculate_and_save_label_frequencies(self, all_labels: List[np.ndarray], name: str) -> Dict[str, float]:
        """Calculates and saves labels frequencies in :attr:`label_info_save_dir`."""
        merged_labels = itertools.chain.from_iterable(all_labels)
        if self.verbose:
            logging.info('Three most popular labels')
        self.label_info_save_dir.mkdir(parents=True, exist_ok=True)
        _, label_frequencies, _ = get_label_stats(
            merged_labels, str(self.label_info_save_dir / f'label_count_{name}.tsv')
        )
        return label_frequencies
    
    def _form_batches(
        self,
        input_ids: List[np.ndarray],
        subtokens_mask: List[np.ndarray],
        punct_labels: List[np.ndarray],
        capit_labels: List[np.ndarray],
    ) -> List[Dict[str, np.ndarray]]:
        """

        Args:
            input_ids: a list of 1D int32 arrays which contain token ids of dataset source
            subtokens_mask: a list of 1D boolean arrays which elements are ``True`` if corresponding token is the
                first token in some word
            punct_labels: a list of 1D int32 arrays which contain encoded punctuation labels
            capit_labels: a list of 1D int32 arrays which contain encoded capitalization labels
            waveforms:  a list of 1D float arrays which contain raw waveforms of audios.
            audio_lengths: a list of 1D int32 arrays which contain length of corresponding audio from `waveforms`
            audio_filepaths: a list of strings which contain paths to audio

        Returns:
            a list of batches. Each batch is a dictionary with items:
              - ``'input_ids'``: a ``np.int32`` numpy array;
              - ``'subtokens_mask'``: a boolean numpy array;
              - ``'punct_labels'``: a ``np.int32`` numpy array;
              - ``'capit_labels'``: a ``np.int32`` numpy array.
            If ``self.add_masks_and_segment_ids_to_batch`` is ``True``, then a batch also contain items
              - ``'segment_ids'``: a ``np.int8`` numpy array;
              - ``'input_mask'``: a boolean numpy array;
              - ``'loss_mask'``: a boolean numpy array.
            If ``waveforms`` is not ``None``, then a batch also contain items
              - ``features``: a ``np.float64`` numpy array.
              - ``features_length`` a ``np.int32`` numpy array.
            If ``audio_filepaths`` is not ``None``, then a natch also contain items
              - ``audio_filepaths`` a list of strings.

            The values of a batch dictionary are numpy arrays of identical shape.
        """
        batches = []

        zipped = list(
            zip(
                input_ids,
                subtokens_mask,
                punct_labels,
                capit_labels,
            )
        )

        for item in zipped:
            batch = {
                "input_ids": item[0],
                "subtokens_mask": item[1],
                "punct_labels": item[2].astype(np.int64),
                "capit_labels": item[3].astype(np.int64),
            }
            batches.append(batch)
        return batches

    # TODO: def _adjust_number_of_batches to check if the lenght of the batch is divisible with th number of batch is multiple of 

    # TODO : Put this in the Default collat class
    def collate_fn(self, batches: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        If ``self.use_bucketing`` set to ``True`` returns zeroth batch from ``batches`` list passed for collating and casts ``'segment_ids'``, ``'punct_labels'``,
        ``'capit_labels'`` to types supported by
        :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationModel`
        or :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationLexicalAudioModel` if ``self.use_audio`` set to ``True``
        All output tensors have shape ``[Batch, Time]``.

        .. warning::
            A ``batch_size`` parameter of a PyTorch data loader and sampler has to be ``1`` if ``self.use_bucketing`` set to ``True``

        Args:
            batches (:obj:`List[Dict[str, np.ndarray]]`): a list containing 1 batch passed for collating

        Returns:
            :obj:`Dict[str, torch.Tensor]`: a batch dictionary with following items (for detailed description of batch
            items see method :meth:`__getitem__`):

              - ``'input_ids'`` (:obj:`torch.Tensor`): :obj:`torch.int32` tensor,
              - ``'subtokens_mask'`` (:obj:`torch.Tensor`): :obj:`torch.bool` tensor,
              - ``'punct_labels'`` (:obj:`torch.Tensor`): :obj:`torch.int64` tensor,
              - ``'capit_labels'`` (:obj:`torch.Tensor`): :obj:`torch.int64` tensor,
              - ``'segment_ids'`` (:obj:`torch.Tensor`): :obj:`torch.int32` tensor,
              - ``'input_mask'`` (:obj:`torch.Tensor`): :obj:`torch.bool` tensor,
              - ``'loss_mask'`` (:obj:`torch.Tensor`): :obj:`torch.bool` tensor.
              - ``'features'`` (:obj:`torch.Tensor`): :obj:`torch.float` tensor.
              - ``'features_length'`` (:obj:`torch.Tensor`): :obj:`torch.long` tensor.
        """
        for batch in batches:
            batch_segment_ids, batch_input_mask, batch_loss_mask = create_masks_and_segment_ids(
                batch['input_ids'],
                batch['subtokens_mask'],
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.ignore_start_end,
                self.ignore_extra_tokens,
            )
            batch['segment_ids'] = torch.as_tensor(batch_segment_ids, dtype=torch.int)
            batch['input_mask'] = torch.as_tensor(batch_input_mask)
            batch['loss_mask'] = torch.as_tensor(batch_loss_mask)
            batch['input_ids'] = torch.as_tensor(batch['input_ids'], dtype=torch.int)
            batch['subtokens_mask'] = torch.as_tensor(batch['subtokens_mask'])
            batch['punct_labels'] = torch.as_tensor(batch['punct_labels'], dtype=torch.int)
            batch['capit_labels'] = torch.as_tensor(batch['capit_labels'], dtype=torch.int)

        segment_ids = pad_sequence([batch['segment_ids'] for batch in batches])
        input_mask = pad_sequence([batch['input_mask'] for batch in batches])
        loss_mask = pad_sequence([batch['loss_mask'] for batch in batches])
        input_ids = pad_sequence([batch['input_ids'] for batch in batches], padding_value=self.tokenizer.pad_token_id)
        subtokens_mask = pad_sequence([batch['subtokens_mask'] for batch in batches], padding_value=False)
        punct_labels = pad_sequence([batch['punct_labels'] for batch in batches], padding_value=0)
        capit_labels = pad_sequence([batch['capit_labels'] for batch in batches], padding_value=0)
        return {
            'input_ids': input_ids.T,
            'subtokens_mask': subtokens_mask.T,
            'punct_labels': punct_labels.T,
            'capit_labels': capit_labels.T,
            'segment_ids': segment_ids.T,
            'input_mask': input_mask.T,
            'loss_mask': loss_mask.T,
        }
            
    def __len__(self) -> int:
        return len(self.batches)
        
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # TODO: add a comment for this code
        return self.batches[idx]

