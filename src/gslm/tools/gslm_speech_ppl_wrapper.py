import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import logging
import argparse
from omegaconf import OmegaConf
from fairseq import utils
from textless.data.speech_encoder import SpeechEncoder
from textless.vocoders.tacotron2.vocoder import TacotronVocoder
from sampler import UnitLanguageModelSampler

log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
logger = logging.getLogger(__name__)

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

        logger.info("=> Done!")
        logger.info("... Loading the vocoder")
        self.resynthesizer = TacotronVocoder.by_name(
            dense_model_name="hubert-base-ls960",
            quantizer_model_name="kmeans",
            vocab_size=vocab_size,
        )
        # move to device and eval mode
        self.device = device
        self.speech_encoder = self.speech_encoder.to(self.device)
        # self.sampler.model = self.sampler.model.to(self.device)
        self.sampler = self.sampler.to(self.device) # make sure the sampler knows the device
        logger.info(f"Sampler model device: {self.sampler.device}")
        self.resynthesizer = self.resynthesizer.to(self.device)
        self.speech_encoder.eval()
        self.sampler.model.eval()
        self.resynthesizer.eval()

        logger.info("=> Done!")
        logger.info("GSLM pipeline initialized!")
    
    @property
    def output_sample_rate(self) -> int:
        return self.resynthesizer.output_sample_rate
    
    @torch.no_grad()
    def extract_raw_units(
        self,
        audio_sample
    ) -> torch.Tensor:
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.to(self.device)
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, np.ndarray):
                raw_audio = torch.Tensor(raw_audio).to(self.device)
            else:
                raw_audio = raw_audio.to(self.device)
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        # get audio units
        encoder_output = self.speech_encoder(raw_audio)
        units = encoder_output['units']
        # print(units.shape)
        # print(raw_audio.shape)
        # assert False, "Debug extract units"
        return units.cpu()


    @torch.no_grad()
    def get_per_token_losses(
        self,
        audio_sample
    ) -> torch.Tensor:
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.to(self.device)
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, np.ndarray):
                raw_audio = torch.Tensor(raw_audio).to(self.device)
            else:
                raw_audio = raw_audio.to(self.device)
        # TODO: add sample rate check
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        # get audio units
        encoder_output = self.speech_encoder(raw_audio)
        units = encoder_output['units']
        # perform deduplication
        input_ids, _durations = torch.unique_consecutive(units, return_counts=True)
        input_ids = input_ids.unsqueeze(0)  # add batch dim (1, seq_len)

        # making training samples!!!
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone() # shift tokens to the left
        labels[:, -1] = -100  # don't predict the last token as it has no next token

        # get unit lm logits
        logits = self.sampler.model(input_ids)[0] 
        # calcuate CE loss
        loss_all_tokens = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1).long(),
            ignore_index=-100,
            reduction='none',
        )
        return {
            "raw_units": units,
            "loss_per_tokens": loss_all_tokens
        }

    @torch.no_grad()
    def generate_continuation_audio(
        self,
        audio_sample,
    ):
        if isinstance(audio_sample, torch.Tensor):
            raw_audio = audio_sample.to(self.device)
        else:
            raw_audio, sr = audio_sample["array"], audio_sample["sampling_rate"]
            if isinstance(raw_audio, np.ndarray):
                raw_audio = torch.Tensor(raw_audio).to(self.device)
            else:
                raw_audio = raw_audio.to(self.device)
        # TODO: add sample rate check
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(0)
        sample = self.speech_encoder(raw_audio)
        units = sample["units"]
        # duration = sample["durations"].sum().item()
        # deduplicate units
        units, durations = torch.unique_consecutive(units, return_counts=True)
        duration = durations.sum().item()
        # generate continuation units
        prefix_duration = self.tokens_framerate * duration
        target_duration = self.tokens_framerate * (
            self.max_length - self.trim_trailing_audio_frames
        )

        unit_str = " ".join(list(map(str, units.tolist())))
        sampled_unit_str = self.sampler.sample([unit_str], **self.sampling_kwargs)[0]

        audio = self.resynthesizer(sampled_unit_str)
        audio = audio[
            : int(
                self.resynthesizer.output_sample_rate
                * (prefix_duration + target_duration)
            )
        ]
        return audio.cpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing_audio_fpath", type=str, default=None)
    parser.add_argument("--language_model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--extract_raw_units", action="store_true")
    args = parser.parse_args()
    # detect device
    device = args.device
    # create model
    model = GslmSpeechPplWrapper(
        language_model_dir=args.language_model_dir,
        seed=None,
        temperature=0.7,
        vocab_size=100,
        device=device,
    )
    print(f"Model Input Sample Rate: {model.input_sample_rate}")
    print(f"Model Output Sample Rate: {model.output_sample_rate}")

    # function for localizing ppl function
    def get_per_token_losses(
        audio_sample: torch.Tensor,
    ) -> torch.Tensor:
        return model.get_per_token_losses(audio_sample)
    # function for localizing generation function
    def generate_continuation_audio(
        prompt_audio: torch.Tensor,
    ) -> torch.Tensor:
        return model.generate_continuation_audio(prompt_audio)
    # function for localizing get_model_features function

    def get_model_features(e):
        # audio is 16000hz, maybe resample 
        positive_sample_loss_results = get_per_token_losses(e["positive_audio"])
        e["positive_sample_tokenwise_loss"] = positive_sample_loss_results["loss_per_tokens"]
        e["positive_sample_raw_units"] = positive_sample_loss_results['raw_units']
        negative_sample_loss_results = get_per_token_losses(e["negative_audio"])
        e["negative_sample_tokenwise_loss"] = negative_sample_loss_results["loss_per_tokens"]
        e["negative_sample_raw_units"] = negative_sample_loss_results['raw_units']
        if "consistency" in e["task"]:
            prompt_sample_loss_results = get_per_token_losses(e["prompt_audio"])
            e["prompt_sample_tokenwise_loss"] = prompt_sample_loss_results["loss_per_tokens"]
            e["prompt_sample_raw_units"] = prompt_sample_loss_results['raw_units']
            positive_continuation_result = get_per_token_losses(e["continuation_audio_positive"])
            negative_continuation_result = get_per_token_losses(e["continuation_audio_negative"])
            e["positive_continuation_tokenwise_loss"] = positive_continuation_result['loss_per_tokens']
            e["positive_continuation_raw_units"] = positive_continuation_result['raw_units']
            e["negative_continuation_tokenwise_loss"] = negative_continuation_result['loss_per_tokens']
            e["negative_continuation_raw_units"] = negative_continuation_result['raw_units']
            generated_audio = generate_continuation_audio(e["prompt_audio"])
            e["model_generated_continuation"] = {"sampling_rate": model.output_sample_rate, "array": generated_audio.squeeze().numpy()}
        
        e["code_frame_rate"] =  50,
        e["code_depth"] =  1
        e["model_sampling_rate"] = model.output_sample_rate,
        e["ppl_sanity"] = int((e["positive_sample_tokenwise_loss"].mean() < e["negative_sample_tokenwise_loss"].mean()).item()) #sanity check if number same as SALMon, would be rerun with other methods
        print("sample correct:",e["ppl_sanity"])
        return e
    
    if args.test_only:
        assert args.testing_audio_fpath is not None, "Please provide testing audio file path for test_only mode."
        # load audio for testing
        audio, sr = torchaudio.load(args.testing_audio_fpath)
        audio = audio.to(device)
        # get per token losses
        per_token_losses = get_per_token_losses(audio)
        print("Per token losses:", per_token_losses[:10], "...", per_token_losses.shape)
        # test generation
        # slice first 3 seconds as prompt
        prompt = int(3.0 * sr)
        prompt_audio = audio[:, :prompt]
        generated_audio = generate_continuation_audio(prompt_audio)
        print("Generated audio shape:", generated_audio.shape)
        # save generated audio
        os.makedirs(args.output_dir, exist_ok=True)
        fid = os.path.basename(args.testing_audio_fpath).split(".")[0]
        gen_fpath = os.path.join(args.output_dir, f"{fid}_contd.wav")
        prompt_fpath = os.path.join(args.output_dir, f"{fid}_prompt.wav")
        torchaudio.save(gen_fpath, generated_audio.cpu().unsqueeze(0), model.output_sample_rate)
        torchaudio.save(prompt_fpath, prompt_audio.cpu(), sr)
        print("Generated audio saved to:", gen_fpath)
        print("Prompt audio saved to:", prompt_fpath)
        exit(0)
    
    if not args.extract_raw_units:
        # try hf salmon dataset
        from datasets import Audio, load_dataset
        splts = ['bg_all_consistency', 'bg_domain_consistency', 'gender_consistency', 'rir_consistency', 'sentiment_consistency', 'speaker_consistency', 'bg_alignment', 'sentiment_alignment']
        # splts = ['bg_alignment', 'sentiment_alignment']
        # splts = ['bg_all_consistency']
        for splt in splts:
            print(splt)
            ds = load_dataset("SpeechPPL/SALMon_with_meta", splt)
            # ds["train"] = ds["train"].select([0])  # for testing purpose, only use 5 samples
            ds = ds.map(get_model_features)
            ds = ds.cast_column("model_generated_continuation", Audio(sampling_rate=model.output_sample_rate))
            # save results
            save_dir = os.path.join(args.output_dir, MODEL_NAME, splt)
            os.makedirs(save_dir, exist_ok=True)
            ds.save_to_disk(save_dir)
            print("Results saved to", save_dir)
            # save 10 prompt contd generation results
            gen_save_dir = os.path.join(save_dir, "generation_examples")
            os.makedirs(gen_save_dir, exist_ok=True)
            # for i, item in enumerate(ds['train']):
            for i, item in enumerate(ds['train'].select([0, 1, 2, 3, 4, 5])):
                if "consistency" in item["task"]:
                    gen_audio = item["model_generated_continuation"]["array"]
                    gen_sr = item["model_generated_continuation"]["sampling_rate"]
                    gen_fpath = os.path.join(gen_save_dir, f"{i}_gen.wav")
                    torchaudio.save(gen_fpath, torch.Tensor(gen_audio).unsqueeze(0), gen_sr)
                    prompt_audio = item["prompt_audio"]["array"]
                    prompt_sr = 16000
                    prompt_fpath = os.path.join(gen_save_dir, f"{i}_prompt.wav")
                    torchaudio.save(prompt_fpath, torch.Tensor(prompt_audio).unsqueeze(0), prompt_sr)
            print("Generation examples saved to", gen_save_dir)
            # push to hub
            ds.push_to_hub(f"SpeechPPL/SALMon_{MODEL_NAME}", config_name=splt)
    else:
        # extract raw units and save
        from datasets import load_dataset, load_from_disk
        splts = ['bg_all_consistency', 'bg_domain_consistency', 'gender_consistency', 'rir_consistency', 'sentiment_consistency', 'speaker_consistency']
        # splts = ['bg_alignment', 'sentiment_alignment']
        local_save_dir = os.path.join(args.output_dir, MODEL_NAME)
        # localize the extract_units function
        def extract_raw_units(audio):
            return model.extract_raw_units(audio)

        def get_model_features(e):
            # audio is 16000hz, maybe resample 
            positive_audio = e.get("positive_audio")
            negative_audio = e.get("negative_audio")
            prompt_audio = e.get("prompt_audio")
            e["postive_sample_raw_units"] = extract_raw_units(positive_audio)
            e["negative_sample_raw_units"] = extract_raw_units(negative_audio)
            if "consistency" in e.get("task", ""):
                e["prompt_sample_raw_units"] = extract_raw_units(prompt_audio)
            return e

        for splt in splts:
            print(f"Loading dataset from {local_save_dir}...")
            ds = load_from_disk(os.path.join(local_save_dir, splt))
            ds = ds.map(get_model_features)
            # save results
            save_dir = os.path.join(args.output_dir, MODEL_NAME, "with_raw_units", splt)
            os.makedirs(save_dir, exist_ok=True)
            ds.save_to_disk(save_dir)
            print("Raw unit results saved to", save_dir)
            ds.push_to_hub(f"SpeechPPL/SALMon_{MODEL_NAME}_with_raw_units", config_name=splt)