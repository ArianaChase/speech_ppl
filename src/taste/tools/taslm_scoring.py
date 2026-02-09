import os
import glob
import argparse
import torch
import torchaudio
import librosa
import numpy as np
from pytorch_lightning import seed_everything
from taste_speech import TasteForCausalLM, TasteProcessor
import json
import csv 
from tqdm import tqdm
import scipy.stats
import time
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
start_time = time.time()

MODEL_NAME="TASLM"
TASLM_INPUT_SAMPLING_RATE = 16000
TASLM_OUTPUT_SAMPLING_RATE = 22050

print(torch.version.cuda)
print(torch.cuda.is_available())

class TaslmSpeechPPLWrapper:
    def __init__(
        self,
        pretrained_model_dir: str,
        attn_implementation: str = "sdpa",
        device: str = "cuda",
    ):
        self.device = device
        self.model = TasteForCausalLM.from_pretrained(
            pretrained_model_dir,
            attn_implementation=attn_implementation,
        )
        #self.model = self.model.to(torch.float32)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = TasteProcessor.from_pretrained(
            pretrained_model_dir
        )
        self.generator = self.processor.get_generator(device=self.device)
        self.generate_kwargs = dict(
            llm_tokenizer=self.processor.llm_tokenizer,
            asr_tokenizer=self.processor.audio_tokenizer,
            extra_words=16,
            text_top_p=0.3,
            taste_top_p=0.0,  # not activated for audio embedding continuation
            text_temperature=0.5,
            repetition_penalty=1.1,
            debug=False,
        )
        # re-register mse loss to avoid batch mean reduction
        self.model.spoken_lm.mse_loss_module = torch.nn.MSELoss(reduction="none")
        self.processor.extract_speech_token_on = False
        self.generate_kwargs = dict(
            llm_tokenizer=self.processor.llm_tokenizer,
            asr_tokenizer=self.processor.audio_tokenizer,
            extra_words=16,
            text_top_p=0.3,
            taste_top_p=0.0, # not activated for audio embedding continuation
            text_temperature=0.5,
            repetition_penalty=1.1,
            debug=True,
        )
    
    def get_audio_sample_and_sr(
        self,
        audio_sample,
    ):
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.cpu().numpy() # taste processor expects numpy array
            sr = 16000  # assume the input audio is always 16kHz
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, torch.Tensor):
                raw_audio = raw_audio.cpu().numpy()
            else:
                raw_audio = raw_audio
        # TODO: add sample rate check
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        if sr != TASLM_INPUT_SAMPLING_RATE:
            # resample to 16kHz
            raw_audio = torchaudio.functional.resample(
                torch.Tensor.from_numpy(raw_audio),
                orig_freq=sr,
                new_freq=TASLM_INPUT_SAMPLING_RATE,
            ).cpu().numpy()
            sr = TASLM_INPUT_SAMPLING_RATE
        return raw_audio, sr
        
    
    @torch.no_grad()
    def get_per_word_losses(
        self,
        audio_sample,
        text=None,
        spk_embed=None,
    ) -> torch.Tensor:
        raw_audio, sr = self.get_audio_sample_and_sr(audio_sample)
        # process audio
        inputs = self.processor(
            audio=raw_audio,
            sampling_rate=sr,
            text=text,
            ref_audio_list=[raw_audio],
            output_text_info=True,
            speaker_embed=spk_embed,
        )
        inputs = inputs.to(device=self.device)
        asr_indices, llm_indices = self.model.extract_vq(
            asr_token_ids=inputs["asr_token_ids"],
            asr_token_lengths=inputs["asr_token_lengths"],
            asr_word_ids=inputs["asr_word_ids"],
            llm_token_ids=inputs["llm_token_ids"],
            llm_token_lengths=inputs["llm_token_lengths"],
            llm_word_ids=inputs["llm_word_ids"],
            audio_features=inputs["audio_features"],
            audio_feature_lengths=inputs["audio_feature_lengths"],
        )
        # manually compute per-token loss
        vq_module = self.model.audio_tower.vq.rvq
        slm_outputs = self.model.spoken_lm(
            llm_indices=llm_indices, 
            llm_token_ids=inputs["llm_token_ids"], 
            llm_token_lengths=inputs["llm_token_lengths"], 
            llm_word_ids=inputs["llm_word_ids"],
            vq_module=vq_module,
        )
        mse_loss = self.model.spoken_lm._calcuate_loss_taste_mse(
            vq_module=vq_module,
            taste_logits=slm_outputs["taste_logits"],
            taste_labels=slm_outputs["taste_labels"],
        )
        # for key, val in slm_outputs.items():
        #     print(f"{key}: {val}")
        #     if isinstance(val, torch.Tensor):
        #         print(f"  shape: {val.shape}")
        # print(f"mse_loss shape: {mse_loss.shape}")
        # print(mse_loss)
        mse_loss_by_words = mse_loss.mean(dim=-1).cpu().numpy()
        # print(f"mse_loss_by_words: {mse_loss_by_words}, len: {len(mse_loss_by_words)}")
        # words = inputs["words"][0]
        # print("words:", words,  len(words))
        return mse_loss_by_words

def create_csv_file(output_dir, model, index): # gslm_001
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

    pbar = tqdm(sorted(os.listdir(root_dir)))

    for files in pbar:
        pbar.set_description(f"Getting per token losses for file: {files}")

        filename = os.path.join(root_dir, files)

        audio, sr = librosa.load(filename, sr=TASLM_INPUT_SAMPLING_RATE)

        per_word_losses = taslm_model.get_per_word_losses(
                audio_sample={"array": audio, "sampling_rate": sr}
            )
        
        
        per_token_losses_tensor = torch.from_numpy(per_word_losses)
        per_word_losses_mean = torch.mean(per_token_losses_tensor)
        with open(output_csv, mode="a", newline="") as csvfile:
            fieldnames = ["Speaker", "Audio filename", "Raw Mean of Per Token Losses", "Normalized Per Token Losses"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({"Speaker": speaker, "Audio filename": os.path.basename(filename), "Raw Mean of Per Token Losses": per_word_losses_mean.item()})


def parse_accuracy_scores(filename):
    accuracy_scores = {}
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            accuracy_scores[os.path.basename(audio_file)] = value["accuracy"]

    return accuracy_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        required=True,
        help="Path to the pretrained TASLM model directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--testing_audio_fpath",
        type=str,
        required=False,
        help="Path to an audio file for testing. If set, the script will conduct simple test using the file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)

    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device)

    # Initialize the wrapper
    taslm_model = TaslmSpeechPPLWrapper(
        pretrained_model_dir=args.pretrained_model_dir,
        device=device,
    )

    # info about the program
    print(f"Language model: {MODEL_NAME}")
    print(f"Model Input Sample Rate: {TASLM_INPUT_SAMPLING_RATE}")
    print(f"Device: {device}")

    # get labels to compare to
    print("Getting labels...")

    score_labels = args.labels_dir
    accuracy_scores = parse_accuracy_scores(score_labels)
    accuracy_scores = dict(sorted(accuracy_scores.items()))
    y = list(accuracy_scores.values())

    # calculating per word losses
    print("Calculating per word losses...")
    output_csv = create_csv_file(args.output_dir, "taste", "001")
    input_dataset = args.dataset_dir
    
    pbar = tqdm(sorted(os.listdir(input_dataset)))
    print(pbar)

    # loop through all directories of the dataset
    counter = 0
    for dirs in pbar:
        #if counter >= 5:
        #    break
        speaker = dirs[7:None]
        if int(speaker) != 1076:
            pbar.set_description(f"Processing speaker: {speaker}")
            dir_path = os.path.join(input_dataset, dirs)
            # get losses for each file in the directory and record in csv
            get_directory_losses(dir_path, output_csv, speaker)
        else:
            pass
        #counter += 1

    # normalization (obsolete)
    scaler = MinMaxScaler()
    output_csv_df = pd.read_csv(output_csv)
    losses = output_csv_df["Raw Mean of Per Token Losses"].values
    losses_reshaped = output_csv_df["Raw Mean of Per Token Losses"].values.reshape(-1,1)

    normalized_col = pd.Series(scaler.fit_transform(losses_reshaped).ravel())
    output_csv_df["Normalized Per Token Losses"] = normalized_col

    output_csv_df.to_csv(output_csv, index=False)

    # get labels to compare to
    score_labels = args.labels_dir
    accuracy_scores = parse_accuracy_scores(score_labels)

    # correlation
    x = losses
    print(len(x))
    print(len(y))
    #print(x)
    #print(y)
    print(f"Correlation value is: {scipy.stats.pearsonr(x, y)}")

    print(f"Program finished executing in {time.time() - start_time} seconds.")

