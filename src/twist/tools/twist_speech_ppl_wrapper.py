import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from transformers import AutoModelForCausalLM
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder
from textless.data.speech_encoder import SpeechEncoder

TWIST_OUTPUT_SAMPLE_RATE = 16000

class TwistSpeechPPLWrapper:
    def __init__(
        self,
        twist_model_pretrained_path,
        dense_model="mhubert-base-25hz",
        quantizer_model="kmeans",
        vocab=500,
        device=None,
    ):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load speech encoder and vocoder
        self.encoder = SpeechEncoder.by_name(
            dense_model_name=dense_model,
            quantizer_model_name=quantizer_model,
            vocab_size=vocab,
            deduplicate=True,
            need_f0=False,
            add_bos_eos=False,
        ).eval().to(self.device)

        self.vocoder = CodeHiFiGANVocoder.by_name(
            dense_model_name=dense_model,
            quantizer_model_name=quantizer_model,
            vocab_size=vocab
        ).eval().to(self.device)

        # build twist unit lm
        self.twist_lm = AutoModelForCausalLM.from_pretrained(twist_model_pretrained_path).to(self.device)
        self.twist_lm.eval()
        self.output_sample_rate = TWIST_OUTPUT_SAMPLE_RATE

    @torch.no_grad()
    def get_per_token_losses(
        self,
        audio_sample
    ) -> torch.Tensor:
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.to(self.device)
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, np.ndarray):
                raw_audio = torch.Tensor(raw_audio).to(self.device)
            else:
                raw_audio = raw_audio.to(self.device)
        # TODO: add sample rate check
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        # get input ids for unit lm
        input_ids = self.encoder(raw_audio)['units'].unsqueeze(0)  # (1, seq_len)
        # prepare labels
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone() # shift tokens to the left
        labels[:, -1] = -100  # don't predict the last token as it has no next token

        # get unit lm logits
        logits = self.twist_lm(input_ids)[0] # skip special tokens
        # calcuate CE loss
        loss_all_tokens = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1).long(),
            ignore_index=-100,
            reduction='none',
        )
        return loss_all_tokens

    @torch.no_grad()
    def generate_continuation_audio(
        self,
        audio_sample,
        offset=None
    ):
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.to(self.device)
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, np.ndarray):
                raw_audio = torch.Tensor(raw_audio).to(self.device)
            else:
                raw_audio = raw_audio.to(self.device)
        # TODO: add sample rate check
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        # get input ids for unit lm
        input_ids = self.encoder(raw_audio)['units'].unsqueeze(0)  # (1, seq_len)
        if offset is None:
            offset = self.twist_lm.config.offset
            input_len= int(input_ids.shape[-1])
            generation_len = int(min(250, 3 * input_len))
        generated_ids = self.twist_lm.generate(offset + input_ids, max_length=generation_len, do_sample=True, temperature=0.8) - offset
        full_generation = self.vocoder(generated_ids, dur_prediction=True)

        return full_generation

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Twist Speech PPL Wrapper Test")
    argparser.add_argument("--twist_model_pretrained_path", type=str, required=True, help="Path to pretrained twist model")
    argparser.add_argument("--input_audio_fpath", type=str, required=True, help="Path to testing input audio file")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    argparser.add_argument("--prompt_duration_sec", type=float, default=None, help="Duration of the prompt in seconds")
    argparser.add_argument("--device", type=str, default=None, help="Device to use, e.g., 'cpu' or 'cuda'")
    args = argparser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # get device
    device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # build model
    model = TwistSpeechPPLWrapper(
        twist_model_pretrained_path=args.twist_model_pretrained_path,
        device=device
    )
    # load audio for testing
    audio, sr = torchaudio.load(args.input_audio_fpath)
    # optionally trim audio
    if args.prompt_duration_sec:
        prompt = int(args.prompt_duration_sec * sr)
        audio = audio[:, :prompt]
    audio = audio.to(device)
    # get per token losses
    per_token_losses = model.get_per_token_losses(audio)
    print("Per token losses:", per_token_losses)
    # try conditional generation
    generated_audio = model.generate_continuation_audio(audio).cpu().unsqueeze(0)
    print("Generated audio shape:", generated_audio.shape)
    # save generated audio to output dir
    input_fid = os.path.basename(args.input_audio_fpath).split('.')[0]
    output_audio_fpath = os.path.join(args.output_dir, f"{input_fid}_contd_gen.wav")
    torchaudio.save(output_audio_fpath, generated_audio, sr)