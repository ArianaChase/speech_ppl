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
        self.model = self.model.to(device=self.device, dtype=torch.bfloat16)

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
        text,
        spk_embed=None,
    ) -> torch.Tensor:
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
        #print("logits: ", slm_outputs["taste_logits"])
        mse_loss = self.model.spoken_lm._calcuate_loss_taste_mse(
            vq_module=vq_module,
            taste_logits=slm_outputs["taste_logits"],
            taste_labels=slm_outputs["taste_labels"],
        )
       
        print("asr word ids: ", inputs["asr_word_ids"])
        mse_loss_by_words = mse_loss.mean(dim=-1).to(torch.float32).cpu().numpy()

        # print(f"mse_loss_by_words: {mse_loss_by_words}, len: {len(mse_loss_by_words)}")
        words = inputs["words"][0]
        print("text: ", text)
        print("words:", words,  len(words))
        return mse_loss_by_words

def create_csv_file(output_dir, name):
    filename = '%s/%s' % (output_dir, name)
    print("Creating csv with file name: ", filename, " ...")

    with open(filename, mode="w") as csvfile:
        fieldnames = ["Speaker", "Audio filename", "Word Index", "Word Text",
                      "Model Loss", "Human Word Accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    return filename

def get_directory_losses(dir, csv_name, spk, labels_dict):
    root_dir = dir
    output_csv = csv_name
    speaker = spk
    skipped_files = []  # track mismatches instead of crashing or silently dropping
    file_count = 0
    file_accuracy_sum = 0

    pbar = tqdm(sorted(os.listdir(root_dir)))

    for files in pbar:
        file_path = os.path.join(root_dir, files)
        filename = os.path.basename(file_path)[0:9]

        pbar.set_description(f"Getting per word losses for file: {files}")

        # O(1) dict lookup instead of scanning a list every time
        human_annotation_obj = labels_dict.get(filename)
        if human_annotation_obj is None:
            skipped_files.append((filename, "no annotation found"))
            continue

        audio, sr = librosa.load(file_path, sr=TASLM_INPUT_SAMPLING_RATE)

        # pass canonical text so TASLM's word segmentation matches the annotation
        per_word_losses = taslm_model.get_per_word_losses(
            audio_sample={"array": audio, "sampling_rate": sr},
            text=human_annotation_obj["text"],
        )

        words = human_annotation_obj["words"]

        # sanity check before trusting the alignment — don't assume, verify
        if len(per_word_losses) != len(words):
            skipped_files.append((
                filename,
                f"length mismatch: model={len(per_word_losses)}, annotation={len(words)}"
            ))
            continue

        # one row per word
        with open(output_csv, mode="a", newline="") as csvfile:
            fieldnames = ["Speaker", "Audio filename", "Word Index", "Word Text",
                          "Model Loss", "Human Word Accuracy"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for idx, (loss_val, word_obj) in enumerate(zip(per_word_losses, words)):
                writer.writerow({
                    "Speaker": speaker,
                    "Audio filename": filename,
                    "Word Index": idx,
                    "Word Text": word_obj["text"],
                    "Model Loss": float(loss_val),
                    "Human Word Accuracy": word_obj["accuracy"],
                })
        
        file_count += 1
        file_accuracy_sum += human_annotation_obj["accuracy"]


    return skipped_files, file_count, file_accuracy_sum

def parse_human_annotations(filename):
    human_scores = {}
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            human_scores[audio_file] = {
                "filename": audio_file,
                "text": value["text"],
                "words": value["words"],
                "accuracy": value["accuracy"],
                "fluency": value["fluency"],
                "prosodic": value["prosodic"],
                "completeness": value["completeness"],
            }
    return human_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str, required=True, help="Path to the pretrained TASLM model directory.")
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

    # calculating per word losses
    print("Calculating per word losses...")
    output_csv = create_csv_file(args.output_dir, "taste_likelihood_001")
    input_dataset = args.dataset_dir
    
    pbar = tqdm(sorted(os.listdir(input_dataset)))
    print(pbar)

    counter = 0
    skipped_files = []
    skipped_file_count = 0
    total_file_count = 0
    passed_accuracy_sum = 0

    # loop through all directories of the dataset
    for dirs in pbar:
        if counter >= 6:
            break
        speaker = dirs[7:None]
        if int(speaker) != 1076:
            pbar.set_description(f"Processing speaker: {speaker}")
            dir_path = os.path.join(input_dataset, dirs)
            # get losses for each file in the directory and record in csv
            skipped, passed_file_count, accuracy_sum = get_directory_losses(dir_path, output_csv, speaker, human_scores)
            skipped_files += skipped
            total_file_count += passed_file_count
            passed_accuracy_sum += accuracy_sum
        else:
            pass
        counter += 1


    output_csv_df = pd.read_csv(output_csv)
    x = output_csv_df["Model Loss"].values
    y = output_csv_df["Human Word Accuracy"].values

    print(f"Skipped files: {skipped_files}")

    skipped_sum = 0
    skipped_count = 0

    for i in range(0, len(skipped_files)):
        if skipped_files[i][1] == "no annotation found":
            print(f"Deleting {skipped_files[i][0]} from skipped_files")
            skipped_files.pop(i)
        else:
            human_annotation_obj = human_scores.get(skipped_files[i][0])
            print(f"Inspecting {skipped_files[i][0]}, accuracy = { human_annotation_obj['accuracy'] }")

            skipped_sum += human_annotation_obj["accuracy"]

            skipped_count += 1

    print(f"Mismatched score mean: {skipped_sum / skipped_count}")
    print(f"Matched score mean: {passed_accuracy_sum / total_file_count}")

    print(f"=== Word-level correlation (pooled across all utterances) ===")
    print(f"Number of word-level pairs: {len(x)}")
    print(f"Correlation value is: {scipy.stats.pearsonr(x, y)}")
    print(f"Skipped {len(skipped_files)} files in total.")
    print(f"Passed {total_file_count} files in total.")


    # Capture and format the finish time
    now = datetime.now()
    finish_time = now.strftime("%m-%d-%Y %H:%M")
    print(f"Date and time at completion: {finish_time}")
    print(f"Program '{args.name}' finished executing in {time.time() - start_time} seconds.")