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

def process_speechocean(input_dataset):

        audio_file_info = []
        spk_count = 0
        ignored_speakers = ["1076"]
        removed = []

        pbar = tqdm(sorted(os.listdir(input_dataset)))

        for spk_dir in pbar:
            spk_count += 1
            speaker = spk_dir[7:None]
            pbar.set_description(f"Processing speaker: {speaker}")
            spk_dir_path = os.path.join(input_dataset, spk_dir)
            for audio_file in os.listdir(spk_dir_path):
                audio_path = os.path.join(spk_dir_path, audio_file)
                filename = os.path.basename(audio_path)[0:9]
                audio_file_info.append({
                    "speaker" : speaker,
                    "filename" : filename,
                    "path" : audio_path
                })
        
        for i in range(len(audio_file_info) - 1, -1, -1):
            if audio_file_info[i]["speaker"] in ignored_speakers:
                removed.append(audio_file_info.pop(i))
                          

        return {
            "processed" : audio_file_info,
            "ignored" : removed,
            "spk_count" : spk_count - len(ignored_speakers)
        }

def clean_transcript(text):
    text = text.upper()  # normalize case
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # strip anything that's not a word char or whitespace
    return cleaned_text

def parse_human_annotations(filename):
    human_scores = {}
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            human_scores[audio_file] = {
                "filename" : audio_file,
                "accuracy" : value["accuracy"],
                "fluency" : value["fluency"],
                "prosodic" : value["prosodic"],
                "completeness" : value["completeness"],
                "words" : value["words"],
                "text" : value["text"]
            }
    return human_scores


def get_losses(dataset, labels_dict, alignments_path, granularity, pooling, norm_dict=None, limit=None):
    '''
    dataset         : dataset object with speaker, filename, and path
    labels_dict     : dictionary of human annotated information, sorted by filename 
    granularity     : word/utterance level
    pooling         : pooling method (max/mean/std) - for utterance only
    norm_dict      : dict for normalization
    '''

    ppl_info = []
    error_log = []
    file_count = 0
    nan_count = 0
    lim = limit if limit != None else len(dataset)

    dataset_cleaned = []

    with open(alignments_path, 'r') as f:
            alignment_list = json.load(f)
        
    
    for sample in dataset:
        for idx in range(len(alignment_list) -1, -1, -1):
            if sample['filename'] == alignment_list[idx]['audio_id']:
                dataset_cleaned.append(sample)
            if alignment_list[idx]['speaker'] == '1076':
                alignment_list.pop(idx)
                
    if len(dataset_cleaned) != len(alignment_list):
        raise Exception(f"Length mismatch between alignments ({len(alignment_list)}) and dataset ({len(dataset_cleaned)})")

    pbar = tqdm(dataset_cleaned)

    for sample in pbar:
        if file_count >= lim:
            break
        
        # info
        speaker = sample["speaker"]
        file_path = sample["path"]
        filename = sample["filename"]
        utterance_ppl_info = []

        pbar.set_description(f"Getting losses for file: {filename}")

        # load audio
        audio, sr = librosa.load(file_path, sr=TASLM_INPUT_SAMPLING_RATE)

        # get human annotations
        human_annotation_obj = labels_dict.get(filename)

        per_word_losses_result = taslm_model.get_per_word_losses(
                audio_sample={"array": audio, "sampling_rate": sr},
                text=human_annotation_obj['text']
            )
        per_word_losses = per_word_losses_result['per_word_losses']
        operated_text = clean_transcript(per_word_losses_result['text'])
        words = operated_text.split()
        auc_threshold = 3

        print(f"WORDS: {words}")

        # safety check
        if len(per_word_losses) != len(words):
            error_log.append(f"At file {filename}, loss tensor length does not align with operated text length")

        if len(per_word_losses) != len(human_annotation_obj['words']):
            error_log.append(f"At file {filename}, loss tensor length does not align with human word scores amt")

        # loss recording
        if granularity == "word":
            if len(words) != len(per_word_losses):
                error_log.append(f"At file {filename}, per_word_losses ({len(per_word_losses)}) do not line up with words ({len(words)}): words are {words}")
                continue
            for idx in range(0, len(per_word_losses)):
                loss = per_word_losses[idx].item()

                # normalization
                # TODO: implement normalization

                word = words[idx]
                freq = word_frequency(word, 'en')
                neg_log_freq = -math.log(freq) if freq > 0 else np.nan  # guard against unknown words
                w_mean = None
                w_std = None

                for bucket, item in norm_dict.items():
                    s = item['freq_range']
                    clean_s = s.strip("()[]")
                    left_str, right_str = clean_s.split(",")
                    left = float(left_str)
                    right = float(right_str)
                    interval = pd.Interval(left, right, closed="right")

                    if neg_log_freq in interval:
                        w_mean = item['mean']
                        w_std = item['std']

                        error_log.append(f"{word} has entered bucket {bucket}, freq is {neg_log_freq}, range is {s}")

                if w_mean != None and w_std != None and w_std > 0:
                    loss_norm = (loss - w_mean) / w_std
                else:
                    loss_norm = np.nan

                ppl_info.append({
                    "speaker" : speaker,
                    "filename" : filename,
                    "label" : words[idx],
                    "auc_label" : 1 if human_annotation_obj['words'][idx]['accuracy'] > auc_threshold else 0,
                    "ppl_loss" : loss,
                    "ppl_loss_norm" : loss_norm,
                    "human_score": human_annotation_obj['words'][idx]['accuracy']
                })

        elif granularity == "utterance":

            loss_pooled = None

            # pooling
            if pooling == "mean":
                loss_pooled = np.mean(per_word_losses)
            elif pooling == "max":
                loss_pooled = np.max(per_word_losses)
            elif pooling == "std":
                loss_pooled = np.std(per_word_losses)
            else:
                raise Exception("Invalid pooling method")

            if np.isnan(loss_pooled):
                nan_count += 1

            ppl_info.append({
                "speaker" : speaker,
                "filename" : filename,
                "label" : operated_text,
                "auc_label" : 1 if human_annotation_obj['accuracy'] > auc_threshold else 0,
                "ppl_loss" : loss_pooled,
                "ppl_loss_norm" : None,
                "human_score": human_annotation_obj['accuracy']
            })
        else:
            raise Exception("Invalid granularity.")

        file_count += 1
            
    with open("/home/u5504709/new_work/speech_ppl/src/taste/tools/error_log", "a") as f:
        for i in error_log:
            f.write(i)
            f.write("\n")

    return {
        "results" : ppl_info,
        "nan_count" : nan_count
    }

def append_to_sheet(
    row_data,
    spreadsheet_name="ICASSP 2026 Experiment Results",
    worksheet_name="main",
    service_account_file="/home/u5504709/new_work/speech_ppl/src/service_account.json"
):
    # Authenticate
    creds = Credentials.from_service_account_file(
        service_account_file,
        scopes=SCOPES
    )

    client = gspread.authorize(creds)

    # Open sheet
    spreadsheet = client.open(spreadsheet_name)
    worksheet = spreadsheet.worksheet(worksheet_name)

    # Append row
    worksheet.append_row(row_data)

    print("Spreadsheet updated successfully.")

def word_level_operation(processed_dataset, human_scores, NORM_DICT_DIR, limit=None, output_dir=None):
    with open(f"{NORM_DICT_DIR}/{MODEL_NAME}_word_none_norm.json", "r") as f:
        norm_dict = json.load(f)

    csv_path = f"{output_dir}/{MODEL_TYPE}_{MODEL_NAME}_word_none_losses.csv"
    
    results = get_losses(
        dataset=processed_dataset, 
        labels_dict=human_scores, 
        alignments_path="/home/u5504709/new_work/speech_ppl/src/mfa/phone_extraction.json",
        granularity="word",
        pooling=None,
        norm_dict=norm_dict,
        limit=limit,
        )

    print(results['results'][0])
    ppl_results = results["results"]

    with open(csv_path, "w") as f:
        fieldnames = ppl_results[0].keys()
        dict_writer = csv.DictWriter(f, fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(ppl_results)
    
    # correlate
    df = pd.DataFrame(ppl_results)
    df.dropna(axis=0, subset=df.columns.drop('ppl_loss_norm'), inplace=True)
    x = df["ppl_loss"]
    y = df["human_score"]

    pcc_result = scipy.stats.pearsonr(x, y)
    pcc_stats = pcc_result.statistic
    pcc_pvalue = pcc_result.pvalue

    y_score = df["ppl_loss"]
    y_true = df["auc_label"]
    if len(np.unique(y_true)) != 1:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = "n/a"

    df_norm = pd.DataFrame(ppl_results)
    df_norm.dropna(axis=0, inplace=True)
    x_norm = df_norm["ppl_loss_norm"]
    y_norm = df_norm["human_score"]

    y_score = df_norm["ppl_loss_norm"]
    y_true = df_norm["auc_label"]
    if len(np.unique(y_true)) != 1:
        auc_norm = roc_auc_score(y_true, y_score)
    else:
        auc_norm = "n/a"

    if len(x_norm) > 2 and len(y_norm) > 2:
        pcc_norm_result = scipy.stats.pearsonr(x, y)
        pcc_norm_stats = pcc_norm_result.statistic
        pcc_norm_pvalue = pcc_norm_result.pvalue
    else:
        pcc_norm_stats = "n/a"
        pcc_norm_pvalue = "n/a"
    
    # Record in CSV
    append_to_sheet([MODEL_TYPE, MODEL_NAME, "word", "n/a", pcc_stats, pcc_pvalue, pcc_norm_stats, pcc_norm_pvalue, auc, auc_norm, "n/a", len(df)])

def utterance_level_operation(processed_dataset, human_scores, limit=None, output_dir=None):

    
    for pool in ["mean", "max", "std"]:
        csv_path = f"{output_dir}/{MODEL_TYPE}_{MODEL_NAME}_utterance_none_losses.csv"

        results = get_losses(
            dataset=processed_dataset, 
            labels_dict=human_scores, 
            alignments_path="/home/u5504709/new_work/speech_ppl/src/mfa/phone_extraction.json",
            granularity="utterance",
            pooling=pool,
            norm_dict=None,
            limit=limit,
            )

        print(results['results'][0])
        ppl_results = results["results"]
        nan_percent = (results["nan_count"] / len(ppl_results)) * 100

        with open(csv_path, "w") as f:
            fieldnames = ppl_results[0].keys()
            dict_writer = csv.DictWriter(f, fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(ppl_results)

        # correlate
        df = pd.DataFrame(ppl_results)
        df.dropna(axis=0, subset=df.columns.drop('ppl_loss_norm'), inplace=True)
        x = df["ppl_loss"]
        y = df["human_score"]

        y_score = df["ppl_loss"]
        y_true = df["auc_label"]
        if len(np.unique(y_true)) != 1:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = "n/a"

        pcc_result = scipy.stats.pearsonr(x, y)
        pcc_stats = pcc_result.statistic
        pcc_pvalue = pcc_result.pvalue

        # Record in CSV
        append_to_sheet([MODEL_TYPE, MODEL_NAME, "utterance", pool, pcc_stats, pcc_pvalue, "n/a", "n/a", auc, "n/a", f"{nan_percent:2f}%", len(df)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str, required=True, help="Path to the pretrained TASLM model directory.",)
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.",)
    parser.add_argument("--testing_audio_fpath", type=str, required=False, help="Path to an audio file for testing. If set, the script will conduct simple test using the file.",)
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.",)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
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

    # get labels to compare to
    score_labels = args.labels_dir
    human_scores = parse_human_annotations(score_labels)

    # process dataset
    input_dataset = args.dataset_dir

    processed = process_speechocean(input_dataset)
    processed_dataset = processed["processed"]
    ignored_samples = processed["ignored"]
    spk_count = processed["spk_count"]
    print(f"Processed {len(processed_dataset)} samples.")

    NORM_DICT_DIR = "/home/u5504709/new_work/speech_ppl/src/gslm/tools/result_dicts"
    OUTPUT_DIR = args.output_dir
    LIMIT = None

    # ============= word-level =================   
    word_level_operation(processed_dataset=processed_dataset, human_scores=human_scores, NORM_DICT_DIR=NORM_DICT_DIR, limit=LIMIT, output_dir=OUTPUT_DIR)
    
    # ======= utterance-level ===========
    utterance_level_operation(processed_dataset=processed_dataset, human_scores=human_scores, limit=LIMIT, output_dir=OUTPUT_DIR)
        
    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")