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

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

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

        # making training samples!!!
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone() # shift tokens to the left
        labels[:, -1] = -100  # don't predict the last token as it has no next token

        # get unit language model logits (Hubert units)
        logits = self.sampler.model(input_ids)[0] # raw predicted scores!! No softmax becaue cross_entropy does it
        
        # calcuate CE loss
        loss_all_tokens = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1).long(),
            ignore_index=-100,
            reduction='none',
        )
        return {
            "logits": logits,
            "loss_all_tokens": loss_all_tokens
        }
    
def create_csv_file(output_dir, name): # gslm_001
    filename = '%s/%s' % (output_dir, name)

    print("Creating csv with file name: ", filename, " ...")

    with open(filename, mode="w") as csvfile:
        fieldnames = ["Speaker", "Audio filename", "Raw Mean of Per Token Losses", "Human Annotation (Accuracy)", "Human Annotation (Fluency)", "Human Annotation (Prosody)", "Human Annotation (Completeness)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return filename

def get_directory_losses(dir, csv_name, spk, labels_list):

    root_dir = dir
    output_csv = csv_name
    speaker = spk

    pbar = tqdm(sorted(os.listdir(root_dir)))

    count = 0

    for files in pbar:
        file_path = os.path.join(root_dir, files)
        filename = os.path.basename(file_path)[0:9]

        pbar.set_description(f"Getting per token losses for file: {filename}")

        audio, sr = torchaudio.load(file_path)
        audio = audio.to(device)

        per_token_losses = get_per_token_losses(audio)["loss_all_tokens"]
        per_token_losses_mean = torch.mean(per_token_losses)

        human_annotation_obj = None

        for obj in labels_list:
            if obj["filename"] == filename:
                human_annotation_obj = obj

                with open(output_csv, mode="a", newline="") as csvfile:
                    fieldnames = ["Speaker", "Audio filename", "Raw Mean of Per Token Losses", "Human Annotation (Accuracy)", "Human Annotation (Fluency)", "Human Annotation (Prosody)", "Human Annotation (Completeness)"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        "Speaker": speaker, 
                        "Audio filename": filename, 
                        "Raw Mean of Per Token Losses": per_token_losses_mean.item(),
                        "Human Annotation (Accuracy)": human_annotation_obj["accuracy"],
                        "Human Annotation (Fluency)": human_annotation_obj["fluency"],
                        "Human Annotation (Prosody)": human_annotation_obj["prosodic"],
                        "Human Annotation (Completeness)": human_annotation_obj["completeness"],
                        })
                
                count += 1

                break
    return count

def parse_human_annotations(filename):
    human_scores = []
    unique_speakers = set()

    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            print(audio_file)
            value = data[audio_file]
            if value["age"] < 18:
                unique_speakers.add(audio_file[1:5])

                human_scores.append({
                    "filename" : audio_file,
                    "accuracy" : value["accuracy"],
                    "fluency" : value["fluency"],
                    "prosodic" : value["prosodic"],
                    "completeness" : value["completeness"]
                })
    return human_scores, unique_speakers

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


    args = parser.parse_args()
    
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
    human_scores,unique_speakers = parse_human_annotations(score_labels)
    human_scores  = sorted(human_scores, key=itemgetter("filename"))
    print(f"Length of human scores: {len(human_scores)}")

    # calculating per token losses

    print("Calculating per token losses...")
    output_csv = create_csv_file(args.output_dir, "gslm_likelihood_accuracy_001")
    input_dataset = args.dataset_dir
    
    pbar = tqdm(sorted(os.listdir(input_dataset)))

    sample_count = 0

    for dirs in pbar:
        # if counter >= 5:
        #     break
        speaker = dirs[7:None]
       
        if int(speaker) != 1076:
            pbar.set_description(f"Processing speaker: {speaker}")
            dir_path = os.path.join(input_dataset, dirs)
            # get losses for each file in the directory and record in csv
            sample_count += get_directory_losses(dir_path, output_csv, speaker, human_scores)
        #counter += 1

    # normalization (obsolete)
    output_csv_df = pd.read_csv(output_csv)
    x = output_csv_df["Raw Mean of Per Token Losses"].values

    def calc_correlation(x, dim):
        if (dim == "accuracy"):
            y = output_csv_df["Human Annotation (Accuracy)"]
        elif (dim == "fluency"):
            y = output_csv_df["Human Annotation (Fluency)"]
        elif (dim == "prosodic"):
            y = output_csv_df["Human Annotation (Prosody)"]
        elif (dim == "completeness"):
            y = output_csv_df["Human Annotation (Completeness)"]
        else:
            print(f"Invalid dimension")
            return
        
        result = scipy.stats.pearsonr(x, y)

        print(f"=== Correlation for dimension {dim} ===")
        print("Correlation x len: ", len(x))
        print("Correlation y len: ", len(y))
        print(f"Correlation value is: {result}")

        return result
    
    accuracy_result = calc_correlation(x, "accuracy")
    fluency_result = calc_correlation(x, "fluency")
    prosody_result = calc_correlation(x, "prosodic")
    completeness_result = calc_correlation(x, "completeness")

    speaker_count = len(unique_speakers)

    # Capture and format the finish time 
    now = datetime.now() 
    finish_time = now.strftime("%m-%d-%Y %H:%M") 
    print(f"Date and time at completion: {finish_time}") 
    duration = time.time() - start_time
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")

    append_to_sheet(["Accuracy-" + args.category, args.name + "_" + str(args.index), finish_time, args.model, MODEL_NAME, speaker_count, sample_count, accuracy_result.statistic, accuracy_result.pvalue, duration])
    append_to_sheet(["Fluency-" + args.category, args.name + "_" + str(args.index), finish_time, args.model, MODEL_NAME, speaker_count, sample_count, fluency_result.statistic, fluency_result.pvalue, duration])
    append_to_sheet(["Prosody-" + args.category, args.name + "_" + str(args.index), finish_time, args.model, MODEL_NAME, speaker_count, sample_count, prosody_result.statistic, prosody_result.pvalue, duration])
    append_to_sheet(["Completeness-" + args.category, args.name + "_" + str(args.index), finish_time, args.model, MODEL_NAME, speaker_count, sample_count, completeness_result.statistic, completeness_result.pvalue, duration])
