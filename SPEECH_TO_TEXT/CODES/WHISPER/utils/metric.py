from datasets import load_metric
import torch

class Metric:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.wer_metric = load_metric("wer")
    def __call__(self, logits, label_ids):
        pred_ids = torch.argmax(logits, axis=-1)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_strs = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_strs = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = self.wer_metric.compute(predictions=pred_strs, references=label_strs)
        return wer
    
    