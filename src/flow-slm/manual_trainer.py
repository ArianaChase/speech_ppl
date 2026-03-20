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

    def forward(self, batch, reduction='token'):
        ids, wavs, wav_len = batch
        wav_len = wav_len.float()
        weighted_combined_utterance_scores = None

        # run pipeline (use eval mode for non-training forward)
        eval_mode = not self.training
        logits, ssl_feats, padding_mask, token_logits, tokens, token_padding_mask = self._run_pipeline(wavs, wav_len, eval_mode)

        flow_loss = self._compute_flow_loss(logits, ssl_feats)
        token_loss = self._compute_token_loss(token_logits, tokens, token_padding_mask, self.training)
        
        # 1. output the raw acoustic/linguistic loss for every token of every utterance
        raw_flow_loss_vals = flow_loss.mean(dim=2) # this is a 2D array, rows = utterances, columns = individual tokens
        if token_loss is not None:
            raw_token_loss_vals = token_loss # this is a 2D array, rows = utterances, columns = individual tokens
        else:
            raw_token_loss_vals = None 

        # 2. utterance-level average token scores
        utterance_flow_loss_avgs = (torch.sum(flow_loss * padding_mask.unsqueeze(-1), dim=1) / torch.sum(padding_mask, dim=1)).mean(dim=1)
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
            "row_flow_losses": raw_flow_loss_vals,
            "utterance_token_losses": utterance_token_loss_avgs, # type: ignore
            "utterance_flow_losses": utterance_flow_loss_avgs,
            "weighted_combined_utterance_scores": weighted_combined_utterance_scores
        }

        return results

def load_wav_as_tensor(filename):
    wav, sr = torchaudio.load(filename)
    if sr != 16000: # Flow-SLM usually expects 16k
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    # Ensure mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0), wav.shape[1]

def get_directory_losses(dir, model, spk):
    root_dir = dir
    speaker = spk

    pbar = tqdm(sorted(os.listdir(root_dir)))
    counter = 0
    speaker_results = []

    for files in pbar:
        if counter >= 20:
            break

        pbar.set_description(f"Getting per token losses for file: {files}")

        filename = os.path.join(root_dir, files)

        wav, wav_len = load_wav_as_tensor(filename)
        batch = (None, wav.unsqueeze(0).to("cuda"), torch.tensor([wav_len]).to("cuda"))  # ids=None
        results = model.forward(batch)
        clean_results = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
            for k, v in results.items()
        }

        speaker_results.append((spk, os.path.basename(filename), clean_results))

        counter += 1

    return speaker_results

    
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", help="Path to dataset folder", default=None)
    parser.add_argument("--output_file", type=str)
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
        conf = yaml.safe_load(f))
    conf = munch.munchify(conf)

    # default k future tokens from config if not provided
    if args.use_k_future_tokens == 0 and hasattr(conf.model, "extra_future_tokens") and conf.model.extra_future_tokens > 0:
        args.use_k_future_tokens = conf.model.extra_future_tokens
 
    # evaluation / prediction only
    print(f"evaluation only, loading {args.ckpt_path}")
    language_modeling = LanguageModeling(args=args, conf=conf)

    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    print(state_dict.keys())


    # defensive cleanup of misnamed keys if present
    for bad_key in ("gsml_pipeline.decoder.stop_token.weight", "gsml_pipeline.decoder.stop_token.bias"):
        if bad_key in state_dict:
            state_dict.pop(bad_key, None)
    try:
        language_modeling.load_state_dict(state_dict)
    except Exception:
        language_modeling.load_state_dict(state_dict, strict=False)

    language_modeling.to("cuda") # CRITICAL
    language_modeling.eval()

    # loop through all directories of the dataset
    dataset = args.data_dir
    pbar = tqdm(sorted(os.listdir(dataset)))
    counter = 0

    with open('%s.json' % (args.output_file), 'w') as f:
        with torch.no_grad():
            for dirs in pbar:
                if counter >= 5:
                   break
                speaker = dirs[7:None]
                if int(speaker) != 1076:
                    pbar.set_description(f"Processing speaker: {speaker}")
                    dir_path = os.path.join(dataset, dirs)
                    # get losses for each file in the directory
                    speaker_data = get_directory_losses(dir_path, language_modeling, speaker)

                    for entry in speaker_data:
                        spk, fname, results = entry
                        output_row = {
                            "speaker": spk,
                            "file": fname,
                            "combined_utterance_score": results["weighted_combined_utterance_scores"].item(), # .item() converts to float
                            "utterance_flow_losses": results["utterance_flow_losses"].item(),
                            "utterance_token_losses": results["utterance_token_losses"].item(),
                        }
                        f.write(json.dumps(output_row) + "\n")
                counter += 1


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
