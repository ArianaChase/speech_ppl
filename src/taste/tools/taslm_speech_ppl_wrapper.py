import os
import glob
import argparse
import torch
import torchaudio
import librosa
import numpy as np
from pytorch_lightning import seed_everything
from taste_speech import TasteForCausalLM, TasteProcessor

MODEL_NAME="TASLM"
TASLM_INPUT_SAMPLING_RATE = 16000
TASLM_OUTPUT_SAMPLING_RATE = 22050


class TaslmSpeechPPLWrapper:
    def __init__(
        self,
        pretrained_model_dir: str,
        attn_implementation: str = "sdpa",
        device: str = "cpu",
    ):
        self.device = device
        self.model = TasteForCausalLM.from_pretrained(
            pretrained_model_dir,
            attn_implementation=attn_implementation,
        )
        self.model = self.model.to(torch.float32)
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
    
    @torch.no_grad()
    def generate_continuation_audio(
        self,
        prompt_audio_sample,
        prompt_text=None,
        prompt_spk_embed=None,
    ):
        raw_audio, sr = self.get_audio_sample_and_sr(prompt_audio_sample)
        inputs = self.processor(
            audio=raw_audio,
            sampling_rate=sr,
            text=prompt_text,
            ref_audio_list=[raw_audio],
            speaker_embed=prompt_spk_embed,
        )
        inputs = inputs.to(device=self.device)
        output = self.model.inference_completion(
            **inputs,
            conditional_mode='audio',
            **self.generate_kwargs,
        )
        slm_speech, slm_sr = self.generator.inference(
            speech_token_ids=output['speech_token_ids'], 
            speech_token_lengths=output['speech_token_lengths'],
            flow_embedding=inputs['speaker_embeds']
        )
        return slm_speech, slm_sr


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
        default="cpu",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pre_extract_features",
        action="store_true",
        help="If set, only use the pre-trained ASR model for generating text transcriptions and the speaker embedding model for speaker embeddings before feeding into TASLM.",
    )
    parser.add_argument(
        "--extract_speech_ppl_results",
        action="store_true",
        help="If set, extract speech PPL results using TASLM on the pre-extracted dataset.",
    )
    parser.add_argument(
        "--run_continuation",
        action="store_true",
        help="If set, run continuation based on pre-extracted + speech ppl results dataset.",
    )
    parser.add_argument(
        "--extract_additional_ppl_results",
        action="store_true",
        help="If set, extract additional PPL results such as negative sample PPL on positive text; Synchronize prompt text with positive text for PPL computation.",
    )
    parser.add_argument(
        "--calculate_sanity_only",
        action="store_true",
        help="If set, only calculate the sanity check accuracy from the final results dataset.",
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    # Initialize the wrapper
    taslm_model = TaslmSpeechPPLWrapper(
        pretrained_model_dir=args.pretrained_model_dir,
        device=args.device,
    )
    # Simple test if audio file is provided
    if args.testing_audio_fpath is not None:
        audio_sample, sr = librosa.load(
            args.testing_audio_fpath, sr=TASLM_INPUT_SAMPLING_RATE
        )
        with torch.no_grad():
            per_token_losses = taslm_model.get_per_word_losses(
                audio_sample={"array": audio_sample, "sampling_rate": sr}
            )
            print("Per-token losses:", per_token_losses)
            print("Done.")
    
    # TODO: start full evaluation on each dataset
    from datasets import Audio, load_dataset, load_from_disk

    def extract_asr_text(audio_sample, return_chunks=False):
        audio_sample, sr = taslm_model.get_audio_sample_and_sr(audio_sample)
        return_timestamps = "word" if return_chunks else False
        asr_result = taslm_model.processor.asr_pipeline(
            {'raw': audio_sample, 'sampling_rate': sr},
            return_timestamps=return_timestamps,
            generate_kwargs={
                'language': 'english',
                'forced_decoder_ids': None,
                'task': 'transcribe'
            },
            batch_size=1,
        )
        asr_text = asr_result['text']
        if return_chunks:
            return asr_result
        return asr_text

    def extract_spk_embed(audio_sample):
        audio_sample, sr = taslm_model.get_audio_sample_and_sr(audio_sample)
        # print(audio_sample.dtype, audio_sample.shape, sr)
        audio_sample = audio_sample.astype(np.float32)
        speaker_embedding = taslm_model.processor._get_speaker_embed(
            taslm_model.processor.speaker_embed_onnx_session,
            [audio_sample],
        )
        return speaker_embedding

    def extract_asr_and_spk_embed(e):
        # extract for positive and negative samples
        e["positive_asr_text"]  = extract_asr_text(e["positive_audio"])
        e["positive_spk_embed"] = extract_spk_embed(e["positive_audio"])
        e["negative_asr_text"]  = extract_asr_text(e["negative_audio"])
        e["negative_spk_embed"] = extract_spk_embed(e["negative_audio"])
        # extract for prompt
        if "consistency" in e["task"]:
            e["prompt_asr_text"]  = extract_asr_text(e["prompt_audio"])
            e["prompt_spk_embed"] = extract_spk_embed(e["prompt_audio"])
        
        return e
    
    def parse_prompt_continuation_asr_text_from_positive_chunks(positive_chunks, prompt_end_s):
        sr = 16000
        prompt_text = ""
        continuation_text = ""
        for chunk in positive_chunks:
            timestamp = chunk['timestamp']
            _start_sec, _end_sec = timestamp
            if _end_sec is None:
                _end_sec = _start_sec + 0.02
            _start_s = _start_sec * sr
            _end_s = _end_sec * sr
            if prompt_end_s <= _end_s:
                if prompt_end_s - _start_s > _end_s - _start_s:
                    prompt_text += chunk['text']
                else:
                    continuation_text += chunk['text']
            else:
                prompt_text += chunk['text']
        return prompt_text, continuation_text

    
    def get_speech_ppl_results(e):
        # compute per-token losses for positive and negative samples
        e["positive_sample_wordlevel_loss"] = taslm_model.get_per_word_losses(
            audio_sample=e["positive_audio"],
            text=e["positive_asr_text"],
            spk_embed=e["positive_spk_embed"],
        )
        e["negative_sample_wordlevel_loss"] = taslm_model.get_per_word_losses(
            audio_sample=e["negative_audio"],
            text=e["negative_asr_text"],
            spk_embed=e["negative_spk_embed"],
        )
        # compute for prompt
        if "consistency" in e["task"]:
            e["prompt_sample_wordlevel_loss"] = taslm_model.get_per_word_losses(
                audio_sample=e["prompt_audio"],
                text=e["prompt_asr_text"],
                spk_embed=e["prompt_spk_embed"],
            )
        
        e["code_frame_rate"] = "word-frequency"
        e["code_depth"] = 4
        e["model_sampling_rate"] = TASLM_OUTPUT_SAMPLING_RATE
        e["ppl_sanity"] = int((e["positive_sample_wordlevel_loss"].mean() < e["negative_sample_wordlevel_loss"].mean()).item())
        # print("sample correct:", e["ppl_sanity"])
        return e

    def get_continuation_results(e):
        if "consistency" in e["task"]:
            slm_speech, slm_sr = taslm_model.generate_continuation_audio(
                prompt_audio_sample=e["prompt_audio"],
                prompt_text=e["prompt_asr_text"],
                prompt_spk_embed=e["prompt_spk_embed"],
            )
            # print(slm_speech.shape, slm_sr)
            assert slm_sr == TASLM_OUTPUT_SAMPLING_RATE
            e["model_generated_continuation"] = {
                "array": slm_speech.squeeze().cpu().numpy(),
                "sampling_rate": slm_sr,
            }
        return e
    
    def get_additional_ppl_results(e):
        # try to get positive text based negative ppl
        positive_asr_result = extract_asr_text(e["positive_audio"], return_chunks=True)
        e["positive_asr_text_old"] = e["positive_asr_text"]
        e["negative_asr_text_old"] = e["negative_asr_text"]
        e["negative_sample_wordlevel_loss_old"] = e["negative_sample_wordlevel_loss"]
        e["positive_sample_wordlevel_loss_old"] = e["positive_sample_wordlevel_loss"]
        # caculate new ones (with consistent text transcriptions)
        e["positive_asr_text"] = positive_asr_result['text']
        e["negative_asr_text"] = positive_asr_result['text']
        e["positive_asr_chunks"] = positive_asr_result['chunks']
        e["positive_sample_wordlevel_loss"] = taslm_model.get_per_word_losses(
            audio_sample=e["positive_audio"],
            text=e["positive_asr_text"],
            spk_embed=e["positive_spk_embed"],
        )
        e["negative_sample_wordlevel_loss"] = taslm_model.get_per_word_losses(
            audio_sample=e["negative_audio"],
            text=e["negative_asr_text"],
            spk_embed=e["negative_spk_embed"],
        )
        if "consistency" in e["task"]:
            e["prompt_asr_text_old"] = e["prompt_asr_text"]
            prompt_asr_text, continuation_asr_text = parse_prompt_continuation_asr_text_from_positive_chunks(
                positive_chunks=positive_asr_result['chunks'],
                prompt_end_s=e["audio_transition_s"]
            )
            e["prompt_sample_wordlevel_loss_old"] = e["prompt_sample_wordlevel_loss"]
            e["prompt_sample_wordlevel_loss"] = taslm_model.get_per_word_losses(
                audio_sample=e["prompt_audio"],
                text=prompt_asr_text,
                spk_embed=e["prompt_spk_embed"],
            )
            # get continuation-only ppl
            e["positive_continuation_wordlevel_loss"] = taslm_model.get_per_word_losses(
                audio_sample=e["continuation_audio_positive"],
                text=continuation_asr_text,
                spk_embed=e["positive_spk_embed"],
            )
            e["negative_continuation_wordlevel_loss"] = taslm_model.get_per_word_losses(
                audio_sample=e["continuation_audio_negative"],
                text=continuation_asr_text,
                spk_embed=e["negative_spk_embed"],
            )
            e["prompt_asr_text"] = prompt_asr_text
            e["continuation_asr_text"] = continuation_asr_text
        # e["ppl_sanity"] = int((e["positive_sample_wordlevel_loss"].mean() < e["negative_sample_wordlevel_loss"].mean()).item())
        e["ppl_sanity_aligned"] = int((e["positive_sample_wordlevel_loss"].mean() < e["negative_sample_wordlevel_loss"].mean()).item())
        return e

    if args.pre_extract_features:
        splits = ['bg_all_consistency', 'bg_domain_consistency', 'gender_consistency', 'rir_consistency', 'sentiment_consistency', 'speaker_consistency']
        # splits = ['bg_alignment', 'sentiment_alignment']
        for splt in splits:
            print(f"Extracting asr and spk embed for split: {splt}")
            ds = load_dataset("SpeechPPL/SALMon_with_meta", splt)
            ds = ds.map(
                extract_asr_and_spk_embed
            )
            save_dir = os.path.join(args.output_dir, MODEL_NAME, "pre_extracted", splt)
            os.makedirs(save_dir, exist_ok=True)
            ds.save_to_disk(save_dir)
            print(f"Saved pre-extracted dataset to {save_dir}")
    if args.extract_speech_ppl_results:
        splits = ['bg_all_consistency', 'bg_domain_consistency', 'gender_consistency', 'rir_consistency', 'sentiment_consistency', 'speaker_consistency']
        # splits = ['bg_alignment', 'sentiment_alignment']
        for splt in splits:
            print(f"Evaluating split: {splt}")
            load_dir = os.path.join(args.output_dir, MODEL_NAME, "pre_extracted", splt)
            ds = load_from_disk(load_dir)
            ds = ds.map(
                get_speech_ppl_results
            )
            save_dir = os.path.join(args.output_dir, MODEL_NAME, "ppl_only_results", splt)
            os.makedirs(save_dir, exist_ok=True)
            ds.save_to_disk(save_dir)
            print(f"Saved results dataset to {save_dir}")
            # compute accuracy
            ppl_sanity = sum(ds['train']['ppl_sanity'])
            print(f"Accuracy on {splt}: {ppl_sanity} / {len(ds['train'])} = {ppl_sanity / len(ds['train']):.4f}")
    if args.run_continuation:
        print("Run continuation based on pre-extracted + speech ppl results ds")
        splits = ['bg_all_consistency', 'bg_domain_consistency', 'gender_consistency', 'rir_consistency', 'sentiment_consistency', 'speaker_consistency']
        # splits = ['bg_alignment', 'sentiment_alignment']
        for splt in splits:
            print(f"Run continuation for split: {splt}")
            load_dir = os.path.join(args.output_dir, MODEL_NAME, "ppl_only_results", splt)
            ds = load_from_disk(load_dir)
            # ds = ds['train'].select([73, 74, 75])
            ds = ds.map(
                get_continuation_results
            )
            ds = ds.cast_column("model_generated_continuation", Audio(sampling_rate=TASLM_OUTPUT_SAMPLING_RATE))
            # load_dir = os.path.join(args.output_dir, MODEL_NAME, "results_with_continuation", splt)
            # ds = load_from_disk(load_dir, keep_in_memory=True)
            # ds = ds.remove_columns("model_generated_continuation")
            # ds = ds.rename_column("continuation_speech", "model_generated_continuation")
            # ds = ds.cast_column("model_generated_continuation", Audio(sampling_rate=TASLM_OUTPUT_SAMPLING_RATE))
            save_dir = os.path.join(args.output_dir, MODEL_NAME, "results_with_continuation", splt)
            os.makedirs(save_dir, exist_ok=True)
            ds.save_to_disk(save_dir)
            print(f"Saved results dataset to {save_dir}")
            ds.push_to_hub(f"SpeechPPL/SALMon_{MODEL_NAME}", config_name=splt)
    if args.extract_additional_ppl_results:
        print("Extract additional PPL results")
        # splits = ['bg_alignment', 'sentiment_alignment', 'bg_all_consistency', 'bg_domain_consistency', 'gender_consistency', 'rir_consistency', 'sentiment_consistency', 'speaker_consistency']
        # splits = ['bg_all_consistency', 'bg_domain_consistency', 'gender_consistency', 'rir_consistency', 'sentiment_consistency', 'speaker_consistency']
        # splits = ['bg_alignment', 'sentiment_alignment']
        splits = ['gender_consistency']
        for splt in splits:
            print(f"Evaluating additional ppl for split: {splt}")
            load_dir = os.path.join(args.output_dir, MODEL_NAME, "results_with_continuation", splt)
            ds = load_from_disk(load_dir)
            ds = ds.map(
                get_additional_ppl_results
            )
            save_dir = os.path.join(args.output_dir, MODEL_NAME, "results", splt)
            os.makedirs(save_dir, exist_ok=True)
            ds.save_to_disk(save_dir)
            ppl_sanity_aligned = sum(ds['train']['ppl_sanity_aligned'])
            print(f"Accuracy on {splt}: {ppl_sanity_aligned} / {len(ds['train'])} = {ppl_sanity_aligned / len(ds['train']):.4f}")
            print(f"Saved additional ppl results dataset to {save_dir}")
            ds.push_to_hub(f"SpeechPPL/SALMon_{MODEL_NAME}", config_name=splt)
    if args.calculate_sanity_only:
        print("Calculate sanity only")
        splits = ['bg_alignment', 'sentiment_alignment', 'bg_all_consistency', 'bg_domain_consistency', 'gender_consistency', 'rir_consistency', 'sentiment_consistency', 'speaker_consistency']
        output_fpath = os.path.join(args.output_dir, MODEL_NAME, "sanity_check_results.txt")
        with open(output_fpath, "w") as fout:
            fout.write("Split\tAccuracy_Aligned\tAccuracy_Separated\n")
            for splt in splits:
                print(f"Calculating sanity for split: {splt}")
                load_dir = os.path.join(args.output_dir, MODEL_NAME, "results", splt)
                ds = load_from_disk(load_dir)
                ppl_sanity_aligned = sum(ds['train']['ppl_sanity_aligned'])
                print(f"Accuracy on {splt} (aligned): {ppl_sanity_aligned} / {len(ds['train'])} = {ppl_sanity_aligned / len(ds['train']):.4f}")
                ppl_sanity = sum(ds['train']['ppl_sanity'])
                print(f"Accuracy on {splt} (separated): {ppl_sanity} / {len(ds['train'])} = {ppl_sanity / len(ds['train']):.4f}")
                fout.write(f"{splt}\t{ppl_sanity_aligned / len(ds['train']):.4f}\t{ppl_sanity / len(ds['train']):.4f}\n")
        print(f"Saved sanity check results to {output_fpath}")




