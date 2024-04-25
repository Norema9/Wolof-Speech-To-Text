from ctypes import Union
from typing import Any
import torch

from base.base_trainer import BaseTrainer
from tqdm import tqdm
from torch.cuda.amp import autocast
from logger.pbar import PBar
from typing import Dict, Union
from model.model import PunctuationCapitalizationModel
from transformers import BertTokenizer


class Trainer(BaseTrainer):
    def __init__(self, 
                dist,
                rank,
                n_gpus: int,
                config,
                resume: bool,
                epochs: int,
                steps_per_epoch: int,
                model: PunctuationCapitalizationModel,
                tokenizer: BertTokenizer,
                compute_metric,
                train_dl,
                val_dl,
                train_sampler,
                val_sampler,
                optimizer,
                scheduler,
                save_dir,
                log_dir,
                gradient_accumulation_steps,
                use_amp,
                max_clip_grad_norm
                ):
        super(Trainer, self).__init__(dist, 
                                        rank, 
                                        config,
                                        resume, 
                                        epochs, 
                                        steps_per_epoch,
                                        model, 
                                        tokenizer,
                                        train_dl,
                                        val_dl,
                                        train_sampler,
                                        val_sampler,
                                        optimizer, 
                                        scheduler,
                                        save_dir, 
                                        log_dir,
                                        use_amp,
                                        gradient_accumulation_steps
                                        )
        self.compute_metric = compute_metric
        self.n_gpus = n_gpus
        self.max_clip_grad_norm = max_clip_grad_norm
        self.stateful_metrics = ["train_loss", "train_lr", "train_grad_norm", "train_punct_accuracy", "train_capit_accuracy", "val_loss", "val_punct_accuracy", "val_capit_accuracy"]

    def get_grad_norm(self, params, scale=1) -> torch.tensor:
        """Compute grad norm given a gradient scale."""
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                param_norm = (p.grad.detach().data / scale).norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def gather(self, value: torch.tensor) -> Any:
        # gather value across devices - https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather
        if value.ndim == 0:
            value = value.clone()[None]
        output_tensors = [value.clone() for _ in range(self.dist.get_world_size())]
        self.dist.all_gather(output_tensors, value)
        return torch.cat(output_tensors, dim=0)

    
    def _train_epoch(self, epoch) -> None:
        self.train_sampler.set_epoch(epoch)
        if self.rank == 0:
            print("Epoch {}: ".format(epoch+1))
            pbar = PBar(self.steps_per_epoch, 10, stateful_metrics = self.stateful_metrics)

        if self.resume_step >= 0 and self.rank == 0:
            print("*****Load previous time steps******")
            resume_pbar = tqdm(total=self.resume_step+1)

        for dl_step, batch in enumerate(self.train_dl):
            if self.resume_step >= 0:
                self.resume_step -= 1
                if self.rank == 0:
                    resume_pbar.update()
                    if self.resume_step < 0:
                        resume_pbar.close()
                continue
            with autocast(enabled=self.use_amp):
                # forward
                self.model.train()
                outputs = self.model.forward(**batch)

                # divide loss by gradient accumulation steps since gradients
                # are accumulated for multiple backward passes in PyTorch
                loss = outputs.loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            (punct_accuracy, 
            capit_accuracy, 
            punct_precision, 
            capit_precision, 
            punct_recall, 
            capit_recall) = torch.tensor(self.compute_metric(outputs.punct_logits.detach(), outputs.capit_logits.detach(), batch['punct_labels'], batch['capit_labels']))
            
            # (punct_accuracy, 
            # capit_accuracy) = torch.tensor(self.compute_metric(outputs.punct_logits.detach(), outputs.capit_logits.detach(), batch['punct_labels'], batch['capit_labels']))
            
            # Optimize step
            if (dl_step + 1) % self.gradient_accumulation_steps == 0 or dl_step == len(self.train_dl) - 1:
                # compute grad norm for monitoring
                grad_norm = self.get_grad_norm(self.model.parameters(), scale = self.scaler.get_scale())

                #gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)

                # update parameters
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                scale_after = self.scaler.get_scale()
                is_overflown = scale_after < scale_before
                if is_overflown:
                    print("\n-----Skip update gradients, encounter overflow-----")
                else:
                    self.scheduler.step()
                
                # Logging
                # average over devices in ddp
                if self.n_gpus > 1:
                    loss = self.gather(loss).mean()
                    punct_accuracy  = self.gather(punct_accuracy).mean()
                    capit_accuracy = self.gather(capit_accuracy).mean()
                    punct_precision  = self.gather(punct_precision).mean()
                    capit_precision = self.gather(capit_precision).mean()
                    punct_recall  = self.gather(punct_recall).mean()
                    capit_recall = self.gather(capit_recall).mean()


                train_logs = {
                    "loss": loss * self.gradient_accumulation_steps,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "grad_norm": grad_norm,
                    "punct_accuracy": punct_accuracy,
                    "capit_accuracy": capit_accuracy,
                    "punct_precision": punct_precision,
                    "capit_precision": capit_precision,
                    "punct_recall": punct_recall,
                    "capit_recall": capit_recall
                }
                train_logs = {k: v.item() if hasattr(v, 'item') else v for k, v in train_logs.items()}

                if self.rank == 0:
                    # write train logs
                    self.writer.update(self.completed_steps, 'Train', train_logs)
                    pbar.update(self.pbar_step + 1, "train_", train_logs)
                    
                # Evaluation
                if (self.completed_steps+1) % self.validation_interval == 0:
                    if self.rank == 0:
                        print("\nValidation is in progress...")
                    self.model.eval()
                    val_logs = self._valid_epoch(self.completed_steps)
                
                    if self.rank == 0:
                        # write val logs
                        self.writer.update(self.completed_steps, 'Validation', val_logs)
                        pbar.update(self.pbar_step+1, "val_", val_logs)

                        # Save best
                        if self._is_best_epoch(val_logs['punct_precision'], val_logs['capit_precision'], save_max_metric_score=self.save_max_metric_score):
                            self._save_checkpoint(epoch, dl_step, is_best_epoch=True)
                        else:
                            self._save_checkpoint(epoch, dl_step, is_best_epoch=False)
                    self.dist.barrier()  # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
                self.pbar_step += 1
                self.completed_steps += 1

        # Reset
        self.pbar_step = 0
            
    def _valid_epoch(self, step) -> Dict[str, Union[Any, float]]:
        self.val_sampler.set_epoch(step)

        # init logs
        val_logs = {
            "loss": 0,
            "punct_accuracy": 0,
            "capit_accuracy": 0,
            "punct_precision": 0,
            "capit_precision": 0,
            "punct_recall": 0,
            "capit_recall": 0
        }

        for batch in tqdm(self.val_dl, total = len(self.val_dl), disable = not self.rank == 0):
            with torch.no_grad():
                with autocast(enabled = self.use_amp):
                    outputs = self.model.forward(**batch)

            val_logs["loss"] += outputs.loss / len(self.val_dl)
            (punct_accur, 
            capit_accur, 
            punct_prec, 
            capit_prec, 
            punct_rec, 
            capit_rec) = torch.tensor(self.compute_metric(outputs.punct_logits.detach(), outputs.capit_logits.detach(), batch['punct_labels'], batch['capit_labels'])) / len(self.val_dl)
            val_logs['punct_accuracy'] += punct_accur
            val_logs['capit_accuracy'] += capit_accur
            val_logs['punct_precision'] += punct_prec
            val_logs['capit_precision'] += capit_prec
            val_logs['punct_recall'] += punct_rec
            val_logs['capit_recall'] += capit_rec

        # average over devices in ddp
        if self.n_gpus > 1:
            val_logs = {k: self.gather(v).mean() for k, v in val_logs.items()}
        val_logs = {k: v.item() if hasattr(v, 'item') else v for k, v in val_logs.items()}
        return val_logs
