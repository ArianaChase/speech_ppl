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
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import csv 
import pandas as pd

start_time = time.time()

MODEL_NAME="TWIST1.3B"

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
            self.device = torch.device("cpu")

        # Load speech encoder and vocoder
        self.encoder = SpeechEncoder.by_name(
            dense_model_name=dense_model,
            quantizer_model_name=quantizer_model,
            vocab_size=vocab,
            deduplicate=False, # set to False but mannually deduplicate later if needed
            need_f0=False,
            add_bos_eos=False,
        ).eval().to(self.device)

        # build twist unit lm
        self.twist_lm = AutoModelForCausalLM.from_pretrained(twist_model_pretrained_path).to(self.device) # this is a text LLM
        self.twist_lm.eval()
    
    @torch.no_grad()
    def get_per_token_losses(
        self,
        audio_sample
    ) -> dict:
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.to(self.device)
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, np.ndarray):
                raw_audio = torch.Tensor(raw_audio).to(self.device)
            else:
                raw_audio = raw_audio.to(self.device)

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)

        # get input ids for unit lm
        units = self.encoder(raw_audio)['units']

        # perform deduplication
        input_ids, _durations = torch.unique_consecutive(units, return_counts=True)
        input_ids = input_ids.unsqueeze(0) + self.twist_lm.config.offset # Speech vocab embeddings live behind the text embeddings, offset by a certain number. this is ensuring that the input ids are not clashing with pretrained text embeddings // (1, seq_len)

        # prepare labels
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone() # shift tokens to the left
        labels[:, -1] = -100  # don't predict the last token as it has no next token

        # get unit lm logits
        logits = self.twist_lm(input_ids)[0]

        # calcuate CE loss
        loss_all_tokens = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1).long(),
            ignore_index=-100,
            reduction='none',
        )
        # return loss_all_tokens
        return {
            'logits': logits,
            'loss_all_tokens': loss_all_tokens
        }

def create_csv_file(output_dir, model, index):
    filename = '%s/%s_%s' % (output_dir, model, index)

    print("Creating csv with file name: ", filename, " ...")

    with open(filename, mode="w") as csvfile:
        fieldnames = ["Speaker", "Audio filename", "Raw Mean of Per Token Losses", "Normalized Per Token Losses"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return filename

def get_directory_losses(dir, csv_name, spk):

    root_dir = dir
    output_csv = csv_name
    speaker = spk

    pbar = tqdm(os.listdir(root_dir))

    for files in pbar:
        pbar.set_description(f"Getting per token losses for file: {files}")

        filename = os.path.join(root_dir, files)

        audio, sr = torchaudio.load(filename)
        audio = audio.to(device)

        per_token_losses = get_per_token_losses(audio)["loss_all_tokens"]
        per_token_losses_mean = torch.mean(per_token_losses)

        with open(output_csv, mode="a", newline="") as csvfile:
            fieldnames = ["Speaker", "Audio filename", "Raw Mean of Per Token Losses", "Normalized Per Token Losses"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({"Speaker": speaker, "Audio filename": os.path.basename(filename), "Raw Mean of Per Token Losses": per_token_losses_mean.item()})

        #print("Filename: ", filename)
        #print("Per token losses (after cross entropy):", per_token_losses[:10], "...", per_token_losses.shape)
        #print(f"Mean of losses: {torch.mean(per_token_losses)}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Twist Speech PPL Wrapper Test")
    argparser.add_argument("--language_model_dir", type=str, required=True, help="Path to pretrained twist model")
    argparser.add_argument("--dataset_dir", type=str, required=True, help="Path to input dataset")
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--device", type=str, default=None, help="Device to use, e.g., 'cpu' or 'cuda'")
   
    args = argparser.parse_args()
    
    # get device
    device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # build model
    model = TwistSpeechPPLWrapper(
        twist_model_pretrained_path=args.language_model_dir,
        device=device
    )
    
    def get_per_token_losses(
        audio_sample: torch.Tensor,
    ) -> dict:
        return model.get_per_token_losses(audio_sample)
    
    print("Calculating per token losses...")
    
    output_csv = create_csv_file(args.output_dir, "twist", "001")
    input_dataset = args.dataset_dir

    pbar = tqdm(os.listdir(input_dataset))

    # loop through all directories of the dataset
    counter = 0
    for dirs in pbar:
        if counter > 1:
            break
        speaker = dirs[7:None]
        pbar.set_description(f"Processing speaker: {speaker}")
        dir_path = os.path.join(input_dataset, dirs)
        # get losses for each file in the directory and record in csv
        get_directory_losses(dir_path, output_csv, speaker)
        counter += 1

    scaler = MinMaxScaler()
    output_csv_df = pd.read_csv(output_csv)
    losses = output_csv_df["Raw Mean of Per Token Losses"].values.reshape(-1,1)

    normalized_col = pd.Series(scaler.fit_transform(losses).ravel())
    output_csv_df["Normalized Per Token Losses"] = normalized_col

    output_csv_df.to_csv(output_csv, index=False)

    print(f"Program finished executing in {time.time() - start_time} seconds.")


    # if args.test_only:
    #     # load audio for testing
    #     audio, sr = torchaudio.load(args.input_audio_fpath)
    #     audio = audio.to(device)

    #     # outputs
    #     logits = model.get_per_token_losses(audio)["logits"]
    #     per_token_losses = model.get_per_token_losses(audio)["loss_all_tokens"]
        
    #     print("Logits (raw, non-softmax prediction scores):", logits[:10], "...", logits.shape)
    #     print("Per token losses (after cross entropy):", per_token_losses[:10], "...", per_token_losses.shape)
    #     print(f"Mean of losses: {torch.mean(per_token_losses)}")
    #     exit(0)
        