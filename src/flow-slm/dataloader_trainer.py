"""Training module for continuous GSLM.

This module contains the main training class LanguageModeling which handles
the training, validation, and prediction steps for the continuous GSLM model.
Refactored to split responsibilities into smaller helper methods, fix a couple
of mode/device issues, and improve readability.
"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import torchaudio
from utils import get_cosine_schedule_with_warmup
from pipeline import GSLMPipeline
from losses import FlowLoss
import argparse
import os
import signal
import yaml
import json
import munch
import sys
from tqdm import tqdm
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from utils import replace_values, writing_output_to_file, SaveAtSpecificStep, select_latest_ckpt
from lightning.pytorch.plugins.environments import SLURMEnvironment
from dataset import SpeechDataModule
from datetime import datetime
import time
start_time = time.time()

class LanguageModeling(pl.LightningModule):
    """Main training class for continuous GSLM.
    
    Refactored: helpers for decoder freezing, loss computation, token loss,
    and metrics. Behavior should remain functionally equivalent.
    """

    def __init__(self, args, conf):
        super().__init__()
        self.args = args
        self.conf = conf
        conf_dict = self.conf.toDict()
        self.save_hyperparameters(conf_dict)

        self.gslm_pipeline = GSLMPipeline(conf, args) # initializing the model

        # build loss_fn (FlowLoss) depending on config
        if self.conf.optimizer.loss_function == "FM":
            token_conditioning = getattr(self.conf.model, "token_conditioning", False)
            if token_conditioning:
                token_emb_dim = getattr(self.conf.model, "token_emb_dim", 0)
                future_conditioning = getattr(self.conf.model, "future_conditioning", False)
                if future_conditioning:
                    extra_future_tokens = getattr(self.conf.model, "extra_future_tokens", 1)
                    z_dim = self.conf.model.decoder_dim + token_emb_dim * extra_future_tokens
                else:
                    z_dim = self.conf.model.decoder_dim + token_emb_dim * self.conf.model.reduction_factor
            else:
                z_dim = self.conf.model.decoder_dim

            null_prob = 0.0 if not hasattr(self.conf.optimizer, "null_prob") else self.conf.optimizer.null_prob
            self.loss_fn = FlowLoss(
                target_dim=self.conf.model.ssl_dim * self.conf.model.reduction_factor,
                z_dim=z_dim,
                net=self.gslm_pipeline.decoder.output_proj,
                sigma_min=self.conf.optimizer.sigma_min,
                t_dist=self.conf.optimizer.t_dist,
                null_prob=null_prob,
            )
        else:
            raise NotImplementedError(f"{self.conf.optimizer.loss_function} not implemented.")

        self.token_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        if not hasattr(self.conf.optimizer, "loss_weight"):
            self.conf.optimizer.loss_weight = 1.0

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        if self.conf.optimizer.name == "AdamW":
            opt = torch.optim.AdamW(
                trainable_params,
                lr=self.conf.optimizer.lr,
                betas=self.conf.optimizer.betas,
                weight_decay=self.conf.optimizer.weight_decay,
                eps=self.conf.optimizer.eps,
            )
        elif self.conf.optimizer.name == "AdamW8bit":
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(
                trainable_params,
                lr=self.conf.optimizer.lr,
                betas=self.conf.optimizer.betas,
                weight_decay=self.conf.optimizer.weight_decay,
                eps=self.conf.optimizer.eps,
                percentile_clipping=self.conf.optimizer.percentile_clipping,
            )
        else:
            raise NotImplementedError(f"{self.conf.optimizer.name} not implemented.")

        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.conf.training.num_warmup_steps,
            num_training_steps=self.conf.training.max_steps,
            min_lr_ratio=self.conf.training.min_lr_ratio,
        )

        lr_scheduler_config = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return lr_scheduler_config

    def _run_pipeline(
        self,
        wavs: torch.Tensor,
        wav_len: torch.Tensor,
        eval_mode: bool,
    ):
        """
        Call the pipeline. If eval_mode is True, temporarily set pipeline.eval()
        and restore previous training flag after call.
        """
        if eval_mode:
            was_training = self.gslm_pipeline.training
            self.gslm_pipeline.eval()
            with torch.no_grad():
                print("====== VALIDATE at run_pipeline =======")
                print(wavs.shape, wav_len)

                out = self.gslm_pipeline(wavs, wav_len)
            if was_training:
                self.gslm_pipeline.train()
            return out
        else:
            return self.gslm_pipeline(wavs, wav_len)

    def _compute_flow_loss(self, logits, ssl_feats):
        """Compute flow loss or return zeros if disabled."""
        if getattr(self.conf.optimizer, "loss_weight", 1.0) > 0:
            return self.loss_fn(logits, ssl_feats)
        return torch.zeros_like(logits)

    def _compute_token_loss(self, token_logits, tokens, token_padding_mask, training: bool):
        if self.conf.optimizer.token_loss_weight <= 0:
            return None

        # multi-future-token handling
        if hasattr(self.conf.model, "extra_future_tokens") and self.conf.model.extra_future_tokens > 1:
            k_future = self.conf.model.extra_future_tokens
            token_losses = token_padding_mask.new_zeros((token_padding_mask.shape[0], token_padding_mask.shape[1])).float()
            token_weight = token_padding_mask.new_zeros((token_padding_mask.shape[0], token_padding_mask.shape[1])).float()

            token_logits_i = torch.chunk(token_logits, k_future, dim=2)
            k_future_tokens = k_future if training else self.args.use_k_future_tokens
            L = token_padding_mask.shape[1]
            for i in range(k_future_tokens):
                logits_i = token_logits_i[i].reshape(-1, token_logits_i[i].shape[-1])
                tokens_i = tokens[:, i : i + L].reshape(-1)
                loss_i = self.token_loss_fn(logits_i, tokens_i).reshape(token_logits_i[i].shape[0], token_logits_i[i].shape[1])
                if self.args.ignore_eos and not training:
                    token_padding_mask_no_eos = (token_padding_mask * (tokens_i.reshape(token_logits_i[i].shape[0], token_logits_i[i].shape[1]) != self.gslm_pipeline.eos_token_index)).float()
                    token_losses += loss_i * token_padding_mask_no_eos
                    token_weight += token_padding_mask_no_eos
                else:
                    token_losses += loss_i * token_padding_mask
                    token_weight += token_padding_mask
            token_weight[token_weight == 0] = 1e-6
            final_token_loss = token_losses / token_weight
        else:
            final_token_loss = self.token_loss_fn(token_logits.reshape(-1, token_logits.shape[-1]), tokens.reshape(-1)).reshape(token_logits.shape[0], token_logits.shape[1])

        return final_token_loss

    def forward(self, batch, reduction="token"):
        ids, wavs, wav_len = batch
        # 1. Is the audio itself NaN?
        if torch.isnan(wavs).any():
            print("CRITICAL: Input audio (wavs) contains NaN")

        wav_len = wav_len.float()
        weighted_combined_utterance_scores = None

        # run pipeline (use eval mode for non-training forward)
        eval_mode = not self.training
        logits, ssl_feats, padding_mask, token_logits, tokens, token_padding_mask = self._run_pipeline(wavs, wav_len, eval_mode)

        # 2. Is the Model outputting NaN?
        if torch.isnan(logits).any():
            print("CRITICAL: Model Logits are NaN")
        if torch.isnan(ssl_feats).any():
            print("CRITICAL: SSL Features are NaN")

        flow_loss = self._compute_flow_loss(logits, ssl_feats)
        token_loss = self._compute_token_loss(token_logits, tokens, token_padding_mask, self.training)
        
        # 1. output the raw acoustic/linguistic loss for every token of every utterance
        raw_flow_loss_vals = flow_loss.mean(dim=2) # this is a 2D array, rows = utterances, columns = individual tokens
        if token_loss is not None:
            raw_token_loss_vals = token_loss # this is a 2D array, rows = utterances, columns = individual tokens
        else:
            raw_token_loss_vals = None 

        # 2. utterance-level average token scores
        utterance_flow_loss_avgs = (torch.sum(flow_loss * padding_mask, dim=1) / torch.sum(padding_mask, dim=1)).mean(dim=1)
        if token_loss is not None:
            # by default, ignore eos token during evaluation
            if self.args.ignore_eos and not self.training:
                eos_index = self.gslm_pipeline.eos_token_index
                L = token_padding_mask.shape[1]
                token_padding_mask = token_padding_mask * (tokens[:, :L].squeeze(dim=2) != eos_index)
            utterance_token_loss_avgs = torch.sum(token_loss * token_padding_mask, dim=1) / torch.sum(token_padding_mask, dim=1)

        # 3. combined and weighted (acoustic + linguistic) utterance-level average token scores
            weighted_combined_utterance_scores = utterance_flow_loss_avgs * self.conf.optimizer.loss_weight + self.conf.optimizer.token_loss_weight * utterance_token_loss_avgs

        
        results = {
            "raw_token_losses": raw_token_loss_vals,
            "raw_flow_losses": raw_flow_loss_vals,
            "utterance_token_losses": utterance_token_loss_avgs, # type: ignore
            "utterance_flow_losses": utterance_flow_loss_avgs,
            "weighted_combined_utterance_scores": weighted_combined_utterance_scores
        }

        return results

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            ids, wavs, wav_len = batch
            results = self.forward(batch, reduction=self.args.reduction)

            raw_token_losses = results["raw_token_losses"]
            raw_flow_losses = results["raw_flow_losses"]
            utterance_token_losses = results["utterance_token_losses"]
            utterance_flow_losses = results["utterance_flow_losses"]
            weighted_combined_utterance_scores = results["weighted_combined_utterance_scores"]

            print(results)


        if raw_token_losses is not None:
            return ids, -raw_token_losses, -raw_flow_losses, -utterance_token_losses, -utterance_flow_losses, -weighted_combined_utterance_scores
        else:
            print("No token losses! Returning only flow losses...")
            return ids, -raw_flow_losses, -utterance_flow_losses
        

    
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--data_dir", help="Path to dataset folder", default=None)
    parser.add_argument("--ckpt_path", type=str, help="GSLM checkpoint, for inference only", default=None)
    parser.add_argument("--conf", help="Path to config file")
    parser.add_argument("--training_data", choices=["MLSEn10k", "MLSEn", "MLSEn+people"], default=None)
    parser.add_argument("--predict_id_file", help="Path to prediction dataset ids")
    parser.add_argument("--prediction_output_dir", help="prediction file path to save")
    parser.add_argument("--reduction", help="reduction approach for prediction", default="utterance")
    parser.add_argument("--ignore_eos", action="store_true", help="ignore eos token for prediction")
    parser.add_argument("--use_k_future_tokens", default=0, type=int, help="use k future tokens for prediction")

    args = parser.parse_args()

    # load config
    with open(args.conf) as f:
        conf = yaml.safe_load(f)
    conf = munch.munchify(conf)

    # default k future tokens from config if not provided
    if args.use_k_future_tokens == 0 and hasattr(conf.model, "extra_future_tokens") and conf.model.extra_future_tokens > 0:
        args.use_k_future_tokens = conf.model.extra_future_tokens
 
    # evaluation / prediction only
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluation only, loading {args.ckpt_path}...")
    language_modeling = LanguageModeling(args=args, conf=conf)
    
    print(f"Loading complete, on device: {language_modeling.device}.")

    print("Loading state dictionary...")
    state_dict = torch.load(args.ckpt_path, map_location="cpu")

    # defensive cleanup of misnamed keys if present
    print("Performing defensive cleanup of misnamed keys...")
    
    if "gslm_pipeline.ssl_model.resample.kernel" in state_dict:
        print("Removing unexpected resample.kernel from state_dict...")
        state_dict.pop("gslm_pipeline.ssl_model.resample.kernel")

    # Defensive cleanup of other known bad keys
    for bad_key in ("gslm_pipeline.decoder.stop_token.weight", "gslm_pipeline.decoder.stop_token.bias"):
        state_dict.pop(bad_key, None)

    # Load with diagnostic info
    missing_keys, unexpected_keys = language_modeling.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print("! WARNING: The following keys are MISSING from the checkpoint:")
        for k in missing_keys:
            print(f"  - {k}")
        print("Missing keys will result in NaNs if you used to_empty() or if they are layer weights.")
    
    if unexpected_keys:
        print(f"! NOTE: Ignored {len(unexpected_keys)} unexpected keys from checkpoint.")

    try:
        language_modeling.load_state_dict(state_dict)
    except Exception:
        language_modeling.load_state_dict(state_dict, strict=True)

    print("State dictionary loaded. Switching to eval mode...")

    language_modeling.eval()

    # preparing dataloader
    precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else 32
    ckpt_dir = os.path.dirname(args.ckpt_path)
    trainer = pl.Trainer(
        accelerator="gpu",
        max_steps=conf.training.max_steps,
        check_val_every_n_epoch=None,
        precision=precision,
        devices="auto",
        accumulate_grad_batches=conf.training.accumulate_grad_batches,
        default_root_dir=ckpt_dir,
        use_distributed_sampler=False,
    )

    print("Starting prediction...")
    data = SpeechDataModule(args, conf)
    data.setup(stage="predict")
    print(data)
    output = trainer.predict(language_modeling, data)
    writing_output_to_file(output, args.prediction_output_dir, token=conf.optimizer.token_loss_weight > 0)

    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")

if __name__ == "__main__":
    print("cuda_available()", torch.cuda.is_available())
    print("cuda_device_count()", torch.cuda.device_count())
    print("torch.cuda.is_bf16_supported()", torch.cuda.is_bf16_supported())
    print("flash_sdp_enabled()", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled()", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled()", torch.backends.cuda.math_sdp_enabled())
    if torch.cuda.is_bf16_supported():
        torch.set_float32_matmul_precision("medium")
    main()
