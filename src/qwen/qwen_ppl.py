import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from tqdm import tqdm
import time
from datetime import datetime
from difflib import SequenceMatcher
from wordfreq import word_frequency
import math
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import scipy.stats
import csv
import librosa
from pathlib import Path
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts import Qwen3TTSTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

qwen = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', torch_dtype=torch.bfloat16)
codec = Qwen3TTSTokenizer.from_pretrained('Qwen/Qwen3-TTS-Tokenizer-12Hz', device_map=str(device))

model = qwen.model
model = qwen.model.to(device)
cfg = model.config
tk = cfg.talker_config
processor = qwen.processor

TOKEN_DURATION = 0.08
NUM_CODE_GROUPS = 16                # columns per frame; check config.json if you swap checkpoints
IGNORE_ID = -100                    # positions we do not want a loss for
BATCH_SIZE = 8

print(device)
print(torch.cuda.is_available())


def process_synth(input_dataset, audio_version, metadata, text_version):

    audio_file_info = []

    pbar = tqdm(os.listdir(input_dataset))
    file_count = 0

    for file_path in pbar:
        if file_count > 1:
            pass
        audio_path = f"{input_dataset}/{file_path}"
        file_metadata = os.path.basename(audio_path).split("_")
        filename = file_metadata[0]
        ver = Path(file_metadata[1]).stem
        if args.set == "setC":
            #print(filename)
            metadata_obj = metadata[metadata["utt_id"] == filename].iloc[0]
        else:
            try:
                metadata_obj = metadata[metadata["stim_id"] == filename].iloc[0]
            except IndexError:
                continue

        if args.set == 'setC' and (metadata_obj['has_foil'] != "yes"):
            continue
        if ver != audio_version:
            continue

        if args.test == "B":
            if args.set == "setC":
                if text_version == "real":
                    text = metadata_obj['actual_text']
                elif text_version == "foil":
                    text = metadata_obj['foil_text']
                elif text_version == "ref":
                    text = metadata_obj['canonical_text']
            else:
                if text_version == "clean":
                    text = metadata_obj['canonical_text']
                else:
                    text = metadata_obj['substituted_text']
        else:
            text = metadata_obj['canonical_text']

        audio_file_info.append({
            "filename" : filename,
            "audio_version" : ver,
            "path" : audio_path,
            "text" : text
        })
        file_count += 1
    return {
        "processed" : audio_file_info,
    }

def is_overlapping(a_start, a_end, b_start, b_end):
    if (a_end >= b_start and a_start <= b_end):
        return True
    else:
        return False

def strip_stress(phone_label):
    if phone_label[-1].isdigit():
        return phone_label[:-1]
    else:
        return phone_label

def build_channels(text_token, speech_token):
    """Lay the text ids and the frame codes out into the two channels the talker expects.
    Returns the text channel, the codec channel, and the index where the first frame sits."""

    L = text_token.size(0)
    C = speech_token.size(0)
    A = 8 + L - 1                   # first audio frame
    N = A + C + 1                   # last slot is end-of-speech

    txt = torch.full((N,), cfg.tts_pad_token_id, dtype=torch.long)
    cdc = torch.zeros(N, dtype=torch.long)

    txt[0:3]   = text_token[:3]
    txt[7]     = cfg.tts_bos_token_id
    txt[8:A-2] = text_token[3:]
    txt[A-2]   = cfg.tts_eos_token_id

    cdc[3:8]   = torch.tensor([tk.codec_nothink_id, tk.codec_think_bos_id,
                               tk.codec_think_eos_id, 0, tk.codec_pad_id])
    cdc[8:A-1] = tk.codec_pad_id
    cdc[A-1]   = tk.codec_bos_id
    cdc[A:A+C] = speech_token[:, 0]
    cdc[A+C]   = tk.codec_eos_token_id

    return txt, cdc, A

def collate_batch(samples):
    """samples = list of dicts, each with txt, cdc, speech_token, frame_start, embedding (all unbatched)"""

    max_len = max(s['txt'].size(0) for s in samples)
    b = len(samples)

    txt = torch.full((b, max_len), cfg.tts_pad_token_id, dtype=torch.long)
    cdc = torch.full((b, max_len), tk.codec_pad_id, dtype=torch.long)
    codes_full = torch.zeros((b, max_len, NUM_CODE_GROUPS), dtype=torch.long)
    frame_mask = torch.zeros((b, max_len), dtype=torch.bool)
    attention_mask = torch.zeros((b, max_len), dtype=torch.long)
    lm_target = torch.full((b, max_len), IGNORE_ID, dtype=torch.long)

    for i, s in enumerate(samples):
        n = s['txt'].size(0)
        A = s['frame_start']
        C = s['speech_token'].size(0)

        txt[i, :n] = s['txt']
        cdc[i, :n] = s['cdc']
        codes_full[i, A:A+C] = s['speech_token']
        frame_mask[i, A:A+C] = True
        attention_mask[i, :n] = 1
        # slot p predicts slot p+1, so the target sits one to the left of the frame it names
        lm_target[i, A-1:A+C] = s['cdc'][A:A+C+1]

    batch = {
        'txt': txt,
        'cdc': cdc,
        'codes_full': codes_full,
        'frame_mask': frame_mask,
        'attention_mask': attention_mask,
        'lm_target': lm_target,
        'embedding': torch.stack([s['embedding'].squeeze(0) for s in samples]),
    }
    return batch

def process_batch(sample_buffer, model, device):
    batch_info = []
    batch = collate_batch(sample_buffer)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        # text channel + codec channel + the true fine codebooks, all summed into one array
        #print(batch['txt'].device)
        emb = model.talker.text_projection(model.talker.model.text_embedding(batch['txt']))

        c = model.talker.model.codec_embedding(batch['cdc'])
        c[:, :3] = 0                                        # nothing in the codec channel yet
        c[:, 6] = batch['embedding'].to(c.dtype)            # slot 6 is reserved for the voice
        emb = emb + c

        for i in range(1, NUM_CODE_GROUPS):
            fine = model.talker.code_predictor.get_input_embeddings()[i-1](batch['codes_full'][:, :, i])
            emb = emb + fine * batch['frame_mask'].unsqueeze(-1)

        out = model.talker(
            inputs_embeds=emb.to(model.dtype),
            attention_mask=batch['attention_mask'],
            use_cache=False
        )
        logits_reshaped = out.logits.float().transpose(1, 2)

        surprisal_per_position = F.cross_entropy(
            logits_reshaped,
            batch['lm_target'].long(),
            ignore_index=IGNORE_ID,
            reduction='none'
        )

    mask = (batch['lm_target'] != IGNORE_ID)

    for i, sample_meta in enumerate(sample_buffer):
        # for each utterance:
        row_mask = mask[i]
        row_surprisal = surprisal_per_position[i][row_mask]
        row_surprisal = row_surprisal[:-1]

        utterance_ppl = row_surprisal.mean().exp().item()
        timestamps_start = [(t * TOKEN_DURATION) for t in range(len(row_surprisal))]
        timestamps_end = [((t + 1) * TOKEN_DURATION) for t in range(len(row_surprisal))]
        timestamped_surprisal = list(zip(row_surprisal.tolist(), timestamps_start, timestamps_end)) # a list of (row_surprisal[idx], t_start, t_end)
        #print(f"row surprisal: {len(row_surprisal)}")
        #print(f"speech_tokens: {sample_meta['speech_token'].size(0)}")
        sample_meta.update({'timestamped_surprisal' : timestamped_surprisal})
        batch_info.append(sample_meta)

    return batch_info

def get_losses(dataset, labels_dict, alignments_path, spk_emb_type, spk_emb_dict, limit=None):
    '''
    dataset         : dataset object with speaker, filename, and path
    labels_dict     : dictionary of human annotated information, sorted by filename
    alignments_path : json file of phone-level alignment boundaries
    granularity     : phone/word/utterance level
    pooling         : pooling method (max/mean/std)
    norm_dict      : dict for normalization
    '''

    ppl_results = []
    file_count = 0
    lim = limit if limit != None else len(dataset)

    pbar = tqdm(dataset)
    sample_buffer = []

    if spk_emb_type == "native_retrieval":
        # NOTE: these dicts were built with CosyVoice's speaker encoder. Regenerate them with
        # model.extract_speaker_embedding before using this branch, the dims do not match.
        emb_arrays = []
        for spk, emb in spk_emb_dict.items():
            print(f"see what's inside the spk emb dict at each speaker: {torch.Tensor(emb).shape}")
            emb_arrays += emb

        emb_arrays = torch.Tensor(np.stack(emb_arrays)).to(device)

    for sample in pbar:
        if file_count >= lim:
            break

        # info
        audio_version = sample['audio_version']
        file_path = sample["path"]
        filename = sample["filename"]
        text = sample['text']
        pbar.set_description(f"Getting per phone losses for file: {filename}")
        #print(text)
        # sample pre-processing
        canonical_text = text
        wav, _ = librosa.load(file_path, sr=24000, mono=True)

        prompt = f"<|im_start|>assistant\n{canonical_text}<|im_end|>\n<|im_start|>assistant\n"
        text_token = processor(text=prompt, return_tensors="pt")["input_ids"][0][:-5]
        text_token = text_token.to(device)
        #print(text_token.device)
        speech_token = codec.encode([file_path]).audio_codes[0].long().cpu()
        spk_embedding = model.extract_speaker_embedding(wav.astype(np.float32), 24000).cpu()

        embedding = None

        if spk_emb_type == "default":
            embedding = spk_embedding
        elif spk_emb_type == "native_retrieval":
            sims = F.cosine_similarity(spk_embedding, emb_arrays, dim=1).cpu()
            idx = int(torch.argmax(sims))
            embedding = emb_arrays[idx].unsqueeze(0)

        elif spk_emb_type == "domestic_retrieval":
            speaker_embs = spk_emb_dict.get(speaker)
            embs = []
            for file, emb in speaker_embs.items():
                if file == filename:
                    continue
                embs.append(emb)

            embedding = torch.Tensor(np.mean(embs, axis=0)).unsqueeze(0)

        txt, cdc, frame_start = build_channels(text_token, speech_token)

        # get losses
        sample_buffer.append({
            'txt': txt,
            'cdc': cdc,
            'speech_token': speech_token,
            'frame_start': frame_start,
            'embedding': embedding,
            'audio_version' : audio_version,
            'file_path' : file_path,
            'filename' : filename,
            'text' : text
        })

        if len(sample_buffer) == BATCH_SIZE:
            batch_losses = process_batch(sample_buffer, model, device)                  # inference and get losses for batch
            for sample in batch_losses:
                for idx, loss in enumerate(sample['timestamped_surprisal']):
                    ppl_results.append({
                        'audio_version' : sample['audio_version'],
                        'file_path' : sample['file_path'],
                        'filename' : sample['filename'],
                        'token_id' : idx,
                        'ppl_loss' : loss[0],
                        'start' : loss[1],
                        'end' : loss[2]
                    })
            sample_buffer = []

        file_count += 1

    if len(sample_buffer) > 0:
        batch_losses = process_batch(sample_buffer, model, device)
        for sample in batch_losses:
            for idx, loss in enumerate(sample['timestamped_surprisal']):
                ppl_results.append({
                    'audio_version' : sample['audio_version'],
                    'file_path' : sample['file_path'],
                    'filename' : sample['filename'],
                    'token_id' : idx,
                    'ppl_loss' : loss[0],
                    'start' : loss[1],
                    'end' : loss[2]
                })

    print(f"FILES RECORDED: {file_count}")

    return {
        "results" : ppl_results,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--alignments_file", required=True)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--test")
    parser.add_argument("--set")

    args = parser.parse_args()

    METADATA_PATH = "/home/ubuntu/speech_ppl/src/stim_final/setB_stimuli_list.csv"
    metadata = pd.read_csv(METADATA_PATH, dtype=str)

    # process dataset
    input_dataset = args.dataset_dir

    if args.set == "setC":
        AUDIO_VERSION = ['real', 'ref']
        TEXT_VERS = ['real', 'ref', 'foil']
    elif args.set == "setB" and args.test == "B":
        AUDIO_VERSION = ['clean', 'sub']
        TEXT_VERS = ['clean', 'sub']
    elif args.set == "setB" and args.test == "A":
        AUDIO_VERSION = ['clean', 'sub', 'dist']
        TEXT_VERS = [None]
    else:
        AUDIO_VERSION = []
        TEXT_VERS = []

    for audio_vers in AUDIO_VERSION:
        for text_vers in TEXT_VERS: 

            processed = process_synth(input_dataset, audio_vers, metadata, text_vers)
            processed_dataset = processed["processed"]
            print(f"Processed {len(processed_dataset)} samples.")

            NORM_DICT_DIR = f"{args.root_dir}/src/metrics/result_dicts"
            SPK_EMB_DIR = f"{args.root_dir}/src/metrics/"
            MODEL_TYPE = "QWEN3TTS"
            MODEL_NAME = f"QWEN3TTS"
            OUTPUT_DIR = args.output_dir
            SPK_EMB_TYPE = "default"

            if args.test == "A":
                csv_path = f"{OUTPUT_DIR}/{MODEL_TYPE}_{MODEL_NAME}_{audio_vers}_{args.set}_per_token_losses.csv"

            elif args.test == "B":
                csv_path = f"{OUTPUT_DIR}/{MODEL_TYPE}_{MODEL_NAME}_{audio_vers}_{text_vers}_{args.set}_per_token_losses.csv"

            if SPK_EMB_TYPE == "native_retrieval":
                dict_path = SPK_EMB_DIR + "libri_spk_dict.json"
                with open(dict_path, "r") as f:
                    spk_emb_dict = json.load(f)
            elif SPK_EMB_TYPE == "domestic_retrieval":
                dict_path = SPK_EMB_DIR + "speechocean_spk_dict.json"
                with open(dict_path, "r") as f:
                    spk_emb_dict = json.load(f)
            else:
                spk_emb_dict = None

            results = get_losses(
                dataset=processed_dataset,
                labels_dict=metadata,
                alignments_path=args.alignments_file,
                spk_emb_type=SPK_EMB_TYPE,
                spk_emb_dict=spk_emb_dict,
                limit=None,
                )

            ppl_results = results["results"]

            with open(csv_path, "w") as f:
                fieldnames = ppl_results[0].keys()
                dict_writer = csv.DictWriter(f, fieldnames)
                dict_writer.writeheader()
                dict_writer.writerows(ppl_results)