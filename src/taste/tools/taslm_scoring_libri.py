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

def process_librispeech(input_dataset):

    audio_file_info = []

    for sample in input_dataset:

        audio_file_info.append({
            "speaker" : sample["speaker_id"],
            "array" : sample["audio"]["array"],
            "sr" : sample["audio"]["sampling_rate"],
            "text" : sample["text"],
            "filename" : sample["id"] 
        }) 

    return audio_file_info

def process_alignments_ds(input_dataset):
    
    alignments = []

    for sample in input_dataset:
        phone_list = []
        word_list = []

        for phone_alignment in sample['phonemes']:
            phone_list.append({
                "start" : phone_alignment['start'],
                "end" : phone_alignment["end"],
                "label" : phone_alignment["phoneme"]
            })

        for word_alignment in sample["words"]:
            word_list.append({
                "start" : word_alignment['start'],
                "end" : word_alignment["end"],
                "label" : word_alignment['word']
            })

        alignments.append({
            "audio_id" : sample['id'],
            "phone_alignment" : phone_list,
            "word_alignment" : word_list
        })
    
    return alignments



def strip_stress(phone_label):
    if phone_label[-1].isdigit():
        return phone_label[:-1]
    else:
        return phone_label

def get_losses(dataset, granularity, limit=None):
    '''
    dataset         : dataset object with speaker, filename, and path
    granularity     : word only
    '''
    result_dict = {}
    file_count = 0
    error_log = []
    nan_count = 0
    lim = limit if limit != None else len(dataset)

    pbar = tqdm(dataset)
    for sample in pbar:
        if file_count >= lim:
            break
        
        # info
        speaker = sample["speaker"]
        filename = sample["filename"]
        pbar.set_description(f"Getting per phone losses for file: {filename}")

        # load audio
        audio = torch.Tensor(librosa.resample(sample["array"], orig_sr=sample['sr'], target_sr=TASLM_INPUT_SAMPLING_RATE))
        sr = TASLM_INPUT_SAMPLING_RATE

        # get losses
        per_word_losses_result = taslm_model.get_per_word_losses(
                audio_sample={"array": audio, "sampling_rate": sr},
                text=sample['text']
            )
        per_word_losses = per_word_losses_result['per_word_losses']
        operated_text = per_word_losses_result['text']
        words = operated_text.split(' ')[1:]

        print(f"WORDS: {words}")

        # safety check
        if len(per_word_losses) != len(words):
            error_log.append(f"At file {filename}, loss tensor length does not align with operated text length")

        # loss recording

        if granularity == "word":
            for idx in range(0, len(per_word_losses)):
                loss = per_word_losses[idx].item()
                word = words[idx]

                freq = word_frequency(word, 'en')
                neg_log_freq = -math.log(freq) if freq > 0 else np.nan  # guard against unknown words
        
                if word in result_dict:
                    result_dict[word]['freq'] = neg_log_freq
                    result_dict[word]['losses'].append(loss)
                else:
                    result_dict[word] = {
                        'freq' : neg_log_freq,
                        'losses' : [loss]
                    }
        else:
            raise Exception("Invalid granularity")
        file_count += 1

    with open("/home/u5504709/new_work/speech_ppl/src/gslm/tools/error_log", "a") as f:
        for i in error_log:
            f.write(i)
            f.write("\n")

    return result_dict

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
    parser.add_argument("--name", type=str, required=True)


    args = parser.parse_args()
    seed_everything(args.seed)
    open('/home/u5504709/new_work/speech_ppl/src/taste/tools/error_log', 'w').close()

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


    # process dataset
    from datasets import load_dataset
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True)
    processed_dataset = process_librispeech(dataset)
    print(f"Processed {len(processed_dataset)} samples.")

    # calculate losses

    GRANULARITY = "word"
    result_dicts_path = "/home/u5504709/new_work/speech_ppl/src/gslm/tools/result_dicts"

    result_dict = get_losses(
        dataset=processed_dataset, 
        granularity=GRANULARITY,
        limit=None,
        )

    # TODO: Implement bucketing here, then create a dictionary with only the buckets

    NUM_BUCKETS = 5

    all_neg_log_freqs = [item['freq'] for word, item in result_dict.items()]
    nan_count = np.isnan(all_neg_log_freqs).sum()
    x_series = pd.Series(all_neg_log_freqs)
    nan_count = x_series.isna().sum()
    all_neg_log_freqs = x_series.dropna().to_numpy() 
    
    bucket_boundaries = pd.qcut(all_neg_log_freqs, q=NUM_BUCKETS)

    print(f"all neg log freqs: {all_neg_log_freqs}, nan count {nan_count}")
    print(f"Boundaries: {bucket_boundaries}, type {type(bucket_boundaries)}")

    bucketed_word_dict = {}

    for idx, bucket in enumerate(np.unique(bucket_boundaries)):
        print(f"Bucket {idx}: {bucket}")
        print(f"Type: {type(bucket)}")

        bucket_losses = []
        bucket_word_count = 0

        for word, info in result_dict.items():
            if info['freq'] in bucket:
                bucket_losses.append(np.nanmean(info['losses'])) 
                bucket_word_count += 1

        bucketed_word_dict[idx] = {
            'freq_range' : str(bucket),
            'losses' : bucket_losses,
            'mean' : np.mean(bucket_losses),
            'std' : np.std(bucket_losses),
            'count' : bucket_word_count
        }
    
    with open(f"{result_dicts_path}/{MODEL_NAME}_word_none_norm.json", "w") as f:
        json.dump(bucketed_word_dict, f)

    
    print(f"File count: {len(processed_dataset)}")

    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")