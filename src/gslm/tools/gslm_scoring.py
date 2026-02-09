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

start_time = time.time()

log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger(__name__)

<<<<<<< HEAD
=======
print(torch.cuda.is_available())  # True if a GPU is detected
print(torch.cuda.device_count())  # Number of GPUs
print(torch.cuda.current_device())  # Index of the current device
print(torch.cuda.get_device_name(0))  # Name of GPU 0

>>>>>>> 3ba7003 (Commit from remote (full correlation works: gslm, twist, taslm))
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
<<<<<<< HEAD

    for files in pbar:
=======
    counter = 0

    for files in pbar:
        if counter >= 20:
            break

>>>>>>> 3ba7003 (Commit from remote (full correlation works: gslm, twist, taslm))
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

<<<<<<< HEAD
=======
        counter += 1
>>>>>>> 3ba7003 (Commit from remote (full correlation works: gslm, twist, taslm))
        #print("Filename: ", filename)
        #print("Per token losses (after cross entropy):", per_token_losses[:10], "...", per_token_losses.shape)
        #print(f"Mean of losses: {torch.mean(per_token_losses)}")

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
    #parser.add_argument("--testing_audio_fpath", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--language_model_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    args = parser.parse_args()
    
    # detect device
<<<<<<< HEAD
    device = args.device
=======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>>>>>> 3ba7003 (Commit from remote (full correlation works: gslm, twist, taslm))
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
    
<<<<<<< HEAD
=======
    # get labels to compare to
    score_labels = args.labels_dir
    accuracy_scores = parse_accuracy_scores(score_labels)
    accuracy_scores = dict(sorted(accuracy_scores.items()))
    print(accuracy_scores)
    y = []
    for key, value in accuracy_scores.items():
        print(key[1:5])
        if key[1:5] != "1076":
            y.append(value)



    #first = dict(list(accuracy_scores.items())[:100])

    #print(first)

>>>>>>> 3ba7003 (Commit from remote (full correlation works: gslm, twist, taslm))
    # calculating per token losses

    print("Calculating per token losses...")
    output_csv = create_csv_file(args.output_dir, "gslm", "001")
    input_dataset = args.dataset_dir
    
    pbar = tqdm(sorted(os.listdir(input_dataset)))
<<<<<<< HEAD
    print(pbar)
=======
>>>>>>> 3ba7003 (Commit from remote (full correlation works: gslm, twist, taslm))

    # loop through all directories of the dataset
    counter = 0
    for dirs in pbar:
<<<<<<< HEAD
        if counter >= 5:
            break
        speaker = dirs[7:None]
        pbar.set_description(f"Processing speaker: {speaker}")
        dir_path = os.path.join(input_dataset, dirs)
        # get losses for each file in the directory and record in csv
        get_directory_losses(dir_path, output_csv, speaker)
        counter += 1
=======
        #if counter >= 5:
        #   break
        speaker = dirs[7:None]
        if int(speaker) != 1076:
            pbar.set_description(f"Processing speaker: {speaker}")
            dir_path = os.path.join(input_dataset, dirs)
            # get losses for each file in the directory and record in csv
            get_directory_losses(dir_path, output_csv, speaker)
        #counter += 1
>>>>>>> 3ba7003 (Commit from remote (full correlation works: gslm, twist, taslm))

    # normalization (obsolete)
    scaler = MinMaxScaler()
    output_csv_df = pd.read_csv(output_csv)
    losses = output_csv_df["Raw Mean of Per Token Losses"].values
    losses_reshaped = output_csv_df["Raw Mean of Per Token Losses"].values.reshape(-1,1)

    normalized_col = pd.Series(scaler.fit_transform(losses_reshaped).ravel())
    output_csv_df["Normalized Per Token Losses"] = normalized_col

    output_csv_df.to_csv(output_csv, index=False)

<<<<<<< HEAD
    # get labels to compare to
    score_labels = args.labels_dir
    accuracy_scores = parse_accuracy_scores(score_labels)

    # correlation
    x = losses
    y = list(accuracy_scores.values())[0:100]
=======
    # correlation
    x = losses
  
>>>>>>> 3ba7003 (Commit from remote (full correlation works: gslm, twist, taslm))
    print(len(x))
    print(len(y))
    print(f"Correlation value is: {scipy.stats.pearsonr(x, y)}")

    print(f"Program finished executing in {time.time() - start_time} seconds.")