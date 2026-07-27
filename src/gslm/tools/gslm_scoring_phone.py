import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import logging
import argparse
import json
import scipy.stats
from omegaconf import OmegaConf
from fairseq import utils
from textless.data.speech_encoder import SpeechEncoder
from textless.vocoders.tacotron2.vocoder import TacotronVocoder
from sampler import UnitLanguageModelSampler
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from operator import itemgetter
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher

# SCOPES = [
#     "https://www.googleapis.com/auth/spreadsheets",
#     "https://www.googleapis.com/auth/drive"
# ]

start_time = time.time()

log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger(__name__)

print(torch.cuda.is_available())  # True if a GPU is detected
print(torch.cuda.device_count())  # Number of GPUs
print(torch.cuda.current_device())  # Index of the current device
print(torch.cuda.get_device_name(0))  # Name of GPU 0

MODEL_NAME="GSLM"
GSLM_INPUT_SAMPLE_RATE = 16000
FIELDNAMES = ["filename", "speaker", "ppl_loss", "human_annotated_accuracy"]

class GslmSpeechPplWrapper:
    def __init__(
        self, 
        language_model_dir: str,
        seed: int = None,
        temperature: float = 0.7,
        vocab_size: int = 100,
        device: str = "cpu",
    ):
        logger.info("Initializing the GSLM pipeline.")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            utils.set_torch_seed(seed)
        self.input_sample_rate = GSLM_INPUT_SAMPLE_RATE
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.tokens_framerate = 0.02  # HuBERT framerate
        self.max_length = 1000
        self.trim_trailing_audio_frames = 200
        self.sampling_kwargs = {
            "temperature": self.temperature,
            "sampling": True,
            "beam": 1,
            "prefix_size": -1,
            "max_len_a": 0.0,
            "max_len_b": self.max_length,
        }
        logger.info("... Loading the language model")
        self.sampler = UnitLanguageModelSampler.from_pretrained(
            language_model_dir,
        )
        logger.info("=> Done!")
        logger.info("... Loading the encoder")

        self.speech_encoder = SpeechEncoder.by_name(
            dense_model_name="hubert-base-ls960",
            quantizer_model_name="kmeans",
            vocab_size=vocab_size,
            need_f0=False,
            deduplicate=False, # set to False to mannually deduplicate later if needed
            f0_normalizer=None,
            f0_quantizer=None,
        )

        # move to device and eval mode
        self.device = device
        self.speech_encoder = self.speech_encoder.to(self.device)
        # self.sampler.model = self.sampler.model.to(self.device)
        self.sampler = self.sampler.to(self.device) # make sure the sampler knows the device
        logger.info(f"Sampler model device: {self.sampler.device}")
        self.speech_encoder.eval()
        self.sampler.model.eval()

        logger.info("=> Done!")
        logger.info("GSLM pipeline initialized!")

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
        
        # turn two channeled audio into 1 channel (just one tensor)
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        
        # get audio units... tokenize, if you will
        encoder_output = self.speech_encoder(raw_audio)
        units = encoder_output['units']
        
        # perform deduplication [4,4,3,3,3,3,3,1,2,2...] -> [4,3,1,2]
        input_ids, _durations = torch.unique_consecutive(units, return_counts=True) # return the durations of each token (how many copies)
        input_ids = input_ids.unsqueeze(0)  # add batch dim (1, seq_len)

        # deal with durations and convert to seconds
        print(f"These are the durations: {_durations}")
        token_ends_frames = torch.cumsum(_durations, dim=0) # [5, 3+5, 1+3+5, ...]
        token_starts_frames = token_ends_frames - _durations # [0, 5, 8, ...]
        t_start = token_starts_frames * self.tokens_framerate  # seconds
        t_end = token_ends_frames * self.tokens_framerate

        print(f"These are the second-based START timestamps of each token in this utterance: {t_start}")

        # making training samples!!!
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone() # shift tokens to the left
        labels[:, -1] = -100  # don't predict the last token as it has no next token

        # get unit language model logits (Hubert units)
        logits = self.sampler.model(input_ids)[0] # raw predicted scores!! No softmax becaue cross_entropy does it
        
        logits_reshaped = logits.reshape(-1, logits.size(-1))

        print(f"There are {len(logits_reshaped)} logits.")
        print(f"There are {len(t_start)} START timstamps aligning with each logit.")
        print(f"There are {len(t_end)} END timstamps aligning with each logit.")

        # calcuate CE loss
        loss_all_tokens = F.cross_entropy(
            logits_reshaped,
            labels.reshape(-1).long(),
            ignore_index=-100,
            reduction='none',
        )

        # merge duration info
        loss_with_timestamps = []
        for i in range(0, len(loss_all_tokens)):
            loss_with_timestamps.append((loss_all_tokens[i].item(), t_start[i].item(), t_end[i].item()))
        
        return {
            "logits": logits,
            "loss_all_tokens": loss_all_tokens,
            "loss_with_timestamps": loss_with_timestamps
        }
    
def create_csv_file(output_dir, name): # gslm_001
    filename = '%s/%s' % (output_dir, name)

    print("Creating csv with file name: ", filename, " ...")

    with open(filename, mode="w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
    
    return filename

def is_overlapping(a_start, a_end, b_start, b_end):
    if (a_end >= b_start and a_start <= b_end):
        return True
    else:
        return False

def get_directory_losses(dir, csv_name, spk, labels_dict, alignments_path):

    root_dir = dir
    output_csv = csv_name
    speaker = spk
    phone_level_ppl_info = []
    file_count = 0
    error_log = []

    with open(alignments_path, 'r') as f:
        alignment_list = json.load(f)

    pbar = tqdm(sorted(os.listdir(root_dir)))

    for files in pbar:
        
        file_path = os.path.join(root_dir, files)
        filename = os.path.basename(file_path)[0:9]
        utterance_ppl_info = []

        pbar.set_description(f"Getting per token losses for file: {filename}")

        audio, sr = torchaudio.load(file_path)
        audio = audio.to(device)

        # external preparation
        alignment_obj = next((item for item in alignment_list if item.get('audio_id') == filename), None)
        phone_alignments = alignment_obj["phone_alignment"]

        human_annotation_obj = labels_dict.get(filename)
        phone_scores = []
        for word_obj in human_annotation_obj["words"]:
            for i in range(0, len(word_obj["phones"])):
                phone_scores.append((word_obj["phones"][i], word_obj["phones-accuracy"][i])) 

        # get ppl_losses per token + timestamps
        losses_with_timestamps = get_per_token_losses(audio)["loss_with_timestamps"]

        # align canonical phonemes and phone alignments
        phone_alignments_labels = [item['label'] for item in phone_alignments]
        phone_scores_labels = [item[0] for item in phone_scores]
        matcher = SequenceMatcher(None, phone_scores_labels, phone_alignments_labels)
        opcodes = matcher.get_opcodes()
        has_error = False
        matched_alignments = []
        matched_scores = []

        for tag, a_idx1, a_idx2, b_idx1, b_idx2 in opcodes:
            if tag == "equal":
                matched_scores.extend(phone_scores[a_idx1:a_idx2])
                matched_alignments.extend(phone_alignments[b_idx1:b_idx2])
        
        phone_alignments = matched_alignments
        phone_scores = matched_scores

        if len(phone_alignments) != len(phone_scores):
            error_log.append(f"Alignment mismatch at file {filename}. {len(phone_alignments)} alignments but {len(phone_scores)} scores.")
            error_log.append(f"{[item['label'] for item in phone_alignments]}\n{[item[0] for item in phone_scores]}")

        # aggregate
        for i in range(0, len(phone_alignments)):
            current_phone = phone_alignments[i]
            #print(f"Inspecting phoneme {i+1}, {current_phone['label']}")

            p_start = current_phone["start"]
            p_end = current_phone["end"]
            phone_loss_sum = 0
            phone_loss_count = 0

            for loss_item in losses_with_timestamps:
                t_start = loss_item[1]
                t_end = loss_item[2]
                if is_overlapping(p_start, p_end, t_start, t_end):
                    phone_loss_count += 1
                    phone_loss_sum += loss_item[0]
            
            phone_loss_mean = phone_loss_sum / phone_loss_count
            utterance_ppl_info.append({
                "speaker" : speaker,
                "filename" : filename,
                "phone" : current_phone['label'],
                "phone_ppl_loss" : phone_loss_mean,
                "phone_human_score": phone_scores[i][1]
            })

        # print(f"Phone level info : {utterance_ppl_info}")
        # print(f"There are {len(utterance_ppl_info)} phone level ppl losses in this utterance.")
        # print(f"There are {len(phone_alignments)} phone alignments.")
        # print(f"There are {len(phone_scores)} pieces of human annotated phone scores.")

        if len(phone_alignments) != len(utterance_ppl_info):
            error_log.append(f"[LOSS/ALIGNMENT MISMATCH] Phone segmentation does not match at file {filename}. {len(phone_alignments)} phone alignments but {len(phone_level_ppl_info)} losses") 
        
        phone_level_ppl_info += utterance_ppl_info
        file_count += 1

    with open("/home/u5504709/new_work/speech_ppl/src/gslm/tools/error_log", "a") as f:
        for i in error_log:
            f.write(i)
            f.write("\n")

    return {
        "phone_level_ppl_info" : phone_level_ppl_info,
        "file_count" : file_count
    }

def parse_human_annotations(filename):
    human_scores = {}
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            print(audio_file)
            value = data[audio_file]
            human_scores[audio_file] = {
                "filename" : audio_file,
                "accuracy" : value["accuracy"],
                "fluency" : value["fluency"],
                "prosodic" : value["prosodic"],
                "completeness" : value["completeness"],
                "words" : value["words"],
            }
    return human_scores

def append_to_sheet(
    row_data,
    spreadsheet_name="Pronunciation Evaluation Results",
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--testing_audio_fpath", type=str, default=None)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--language_model_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--alignments", type=str)

    args = parser.parse_args()
    
    open('/home/u5504709/new_work/speech_ppl/src/gslm/tools/error_log', 'w').close()


    # detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model
    model = GslmSpeechPplWrapper(
        language_model_dir=args.language_model_dir,
        seed=None,
        temperature=0.7,
        vocab_size=100,
        device=device,
    )

    # info about the program
    print(f"Language model: {MODEL_NAME}")
    print(f"Model Input Sample Rate: {model.input_sample_rate}")
    print(f"Device: {device}")

    # function for localizing ppl function
    def get_per_token_losses(
        audio_sample: torch.Tensor,
    ) -> dict:
        return model.get_per_token_losses(audio_sample)
    
    # get labels to compare to
    score_labels = args.labels_dir
    human_scores = parse_human_annotations(score_labels)

    # calculating per token losses

    print("Calculating per token losses...")
    output_csv = create_csv_file(args.output_dir, "gslm_ppl_phone")
    input_dataset = args.dataset_dir
    
    pbar = tqdm(sorted(os.listdir(input_dataset)))

    # loop through all directories of the dataset
    counter = 0
    file_count = 0
    phone_level_info = []

    for dirs in pbar:
        if counter >= 20:
           break
        speaker = dirs[7:None]
        if int(speaker) != 1076:
            pbar.set_description(f"Processing speaker: {speaker}")
            dir_path = os.path.join(input_dataset, dirs)
            # get losses for each file in the directory and record in csv
            result =  get_directory_losses(dir_path, output_csv, speaker, human_scores, args.alignments)
            file_count += result["file_count"]
            phone_level_info += result["phone_level_ppl_info"]
        counter += 1

    # correlate
    df = pd.DataFrame(phone_level_info)
    print(df.head())

    x = df["phone_ppl_loss"]
    y = df["phone_human_score"]
    result = scipy.stats.pearsonr(x, y)

    print(f"Speaker count: {counter}")
    print(f"File count: {file_count}")
    print(f"Correlation: {result}")

    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    duration = time.time() - start_time
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")

    # append_to_sheet(["Accuracy-" + args.category, args.name + "_" + str(args.index), finish_time, args.model, MODEL_NAME, accuracy_result.statistic, accuracy_result.pvalue, duration])
    # append_to_sheet(["Fluency-" + args.category, args.name + "_" + str(args.index), finish_time, args.model, MODEL_NAME, fluency_result.statistic, fluency_result.pvalue, duration])
    # append_to_sheet(["Prosody-" + args.category, args.name + "_" + str(args.index), finish_time, args.model, MODEL_NAME, prosody_result.statistic, prosody_result.pvalue, duration])
    # append_to_sheet(["Completeness-" + args.category, args.name + "_" + str(args.index), finish_time, args.model, MODEL_NAME, completeness_result.statistic, completeness_result.pvalue, duration])
