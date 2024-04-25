from datasets import load_metric
import torch
import numpy as np

class Metric:
    def __init__(self, processor):
        self.processor = processor
        self.wer_metric = load_metric("wer")
    def __call__(self, logits, labels):
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        
        if type(self.processor).__name__ == "Wav2Vec2ProcessorWithLM":
            pred_strs    = self.processor.batch_decode(logits.numpy()).text
        else:   
            pred_ids = np.argmax(logits, axis=-1)
            pred_strs = self.processor.batch_decode(pred_ids)
        
        label_strs = self.processor.tokenizer.batch_decode(labels, group_tokens=False)
        
        wer = self.wer_metric.compute(predictions=pred_strs, references=label_strs)
        return wer