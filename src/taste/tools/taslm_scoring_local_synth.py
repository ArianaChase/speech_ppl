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
from datetime import datetime
from operator import itemgetter
import gspread
from google.oauth2.service_account import Credentials
from wordfreq import word_frequency
import math
import os
import re
from pathlib import Path
from sklearn.metrics import roc_auc_score
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
start_time = time.time()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

MODEL_TYPE="TASLM"
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
        self.model = self.model.to(device=self.device, dtype=torch.bfloat16)

        self.model.eval()
        self.processor = TasteProcessor.from_pretrained(
            pretrained_model_dir, asr_on=False
        )
        self.generator = self.processor.get_generator(device=self.device)
        self.generate_kwargs = dict(
            llm_tokenizer=self.processor.llm_tokenizer,
            asr_tokenizer=self.processor.audio_tokenizer,
            asr_on=False,
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
        text,
        spk_embed=None,
    ) -> dict:
        raw_audio, sr = self.get_audio_sample_and_sr(audio_sample)
        # process audio

        # If spk_embed is provided externally, ensure it is bf16
        if spk_embed is not None and torch.is_floating_point(spk_embed):
            spk_embed = spk_embed.to(device=self.device, dtype=torch.bfloat16)

        inputs = self.processor(
            audio=raw_audio,
            sampling_rate=sr,
            text=text,
            ref_audio_list=[raw_audio],
            output_text_info=True,
            speaker_embed=spk_embed,
        )

        # Move to device and cast ONLY floating point tensors to bfloat16
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    inputs[k] = v.to(device=self.device, dtype=torch.bfloat16)
                else:
                    inputs[k] = v.to(device=self.device) # Keep IDs as Integers/Long

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

        mse_loss_by_words = mse_loss.mean(dim=-1).to(torch.float32).cpu().numpy()

        # print(f"mse_loss_by_words: {mse_loss_by_words}, len: {len(mse_loss_by_words)}")
        words = self.processor.llm_tokenizer.decode(inputs["llm_token_ids"][0])  
        print(f"Text given: {text}")

        # print("words:", words,  len(words))
        return {
            "per_word_losses" : mse_loss_by_words,
            "text" : words
        }

def create_csv_file(output_dir, name): # gslm_001
    filename = '%s/%s' % (output_dir, name)

    print("Creating csv with file name: ", filename, " ...")

    with open(filename, mode="w") as csvfile:
        fieldnames = ["Speaker", "Audio filename", "Raw Mean of Per Token Losses", "Human Annotation (Accuracy)", "Human Annotation (Fluency)", "Human Annotation (Prosody)", "Human Annotation (Completeness)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return filename

def process_synth(input_dataset, audio_version, metadata):

    audio_file_info = []

    pbar = tqdm(os.listdir(input_dataset))

    for file_path in pbar:
        audio_path = f"{input_dataset}/{file_path}"
        file_metadata = os.path.basename(audio_path).split("_")            
        filename = file_metadata[0]
        ver = Path(file_metadata[1]).stem

        metadata_obj = metadata[metadata["stim_id"] == filename].iloc[0]

        if ver != audio_version:
            continue
        
        text = metadata_obj['canonical_text']

        audio_file_info.append({
            "filename" : filename,
            "audio_version" : ver,
            "path" : audio_path,
            "text" : text
        })
    return {
        "processed" : audio_file_info,
    }

def clean_transcript(text):
    text = text.upper()  # normalize case
    cleaned_text = re.sub(r'[^\w\s]', '', text) # strip anything that's not a word char or whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def get_losses(dataset, labels_dict, alignments_path, limit=None):
    '''
    dataset         : dataset object with speaker, filename, and path
    alignments_path : to filter the dataset to those that have alignments to compare to downstream
    labels_dict     : to extract canonical text
    '''

    per_word_losses_final = []
    file_count = 0
    lim = limit if limit != None else len(dataset)

    pbar = tqdm(dataset)

    for sample in pbar:
        if file_count >= lim:
            break
        
        # info
        file_path = sample["path"]
        filename = sample["filename"]
        ver = sample["audio_version"]

        pbar.set_description(f"Getting losses for file: {filename}")

        # load audio
        audio, sr = librosa.load(file_path, sr=TASLM_INPUT_SAMPLING_RATE)

        per_word_losses_result = taslm_model.get_per_word_losses(
                audio_sample={"array": audio, "sampling_rate": sr},
                text=clean_transcript(sample['text'])
            )
        per_word_losses = per_word_losses_result['per_word_losses']
        operated_text = clean_transcript(per_word_losses_result['text'])
        words = operated_text.split()

        print(f"LENGTH OF LOSSES {len(per_word_losses)}")
        print(f"LENGTH OF WORDS {len(words)} {operated_text}")

        for idx, loss in enumerate(per_word_losses):
            per_word_losses_final.append({
                "audio_version" : ver,
                "file_path" : file_path,
                "filename" : filename,
                "word_text" : words[idx],
                "token_id" : idx,
                "ppl_loss" : loss,
            })

        file_count += 1
            

    return {
        "results" : per_word_losses_final
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str, required=True, help="Path to the pretrained TASLM model directory.",)
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.",)
    parser.add_argument("--testing_audio_fpath", type=str, required=False, help="Path to an audio file for testing. If set, the script will conduct simple test using the file.",)
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.",)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()
    seed_everything(args.seed)
    open(f'{args.root_dir}/src/taste/tools/error_log', 'w').close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device)

    # Initialize the wrapper
    taslm_model = TaslmSpeechPPLWrapper(
        pretrained_model_dir=args.pretrained_model_dir,
        device=device,
    )

    METADATA_PATH = "/home/ubuntu/speech_ppl/src/stim_final/setB_stimuli_list.csv"
    metadata = pd.read_csv(METADATA_PATH, dtype=str)

    # info about the programF
    print(f"Language model: {MODEL_NAME}")
    print(f"Model Input Sample Rate: {TASLM_INPUT_SAMPLING_RATE}")
    print(f"Device: {device}")

     # process dataset
    input_dataset = args.dataset_dir
    AUDIO_VERSION = "sub"
    processed = process_synth(input_dataset, AUDIO_VERSION, metadata)
    processed_dataset = processed["processed"]
    print(f"Processed {len(processed_dataset)} samples.")

    NORM_DICT_DIR = f"{args.root_dir}/src/gslm/tools/result_dicts"
    SERVICE_ACCOUNT = f"{args.root_dir}/src/service_account.json"
    OUTPUT_DIR = args.output_dir

    csv_path = f"{OUTPUT_DIR}/{MODEL_TYPE}_{MODEL_NAME}_{AUDIO_VERSION}_setB_per_word_losses.csv"   

        
    results = get_losses(
        dataset=processed_dataset, 
        alignments_path=f"{args.root_dir}/src/mfa/phone_extraction.json",
        labels_dict=metadata,
        limit=None,
        )

    ppl_results = results["results"]

    with open(csv_path, "w") as f:
        fieldnames = ppl_results[0].keys()
        dict_writer = csv.DictWriter(f, fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(ppl_results)
        
    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")