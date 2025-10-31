from shutil import copyfile
from tqdm import tqdm
from pytorch_lightning import seed_everything
import torchaudio
import argparse
import glob
import os

from taste_speech import TasteConfig, TasteForCausalLM, TasteProcessor

argparser = argparse.ArgumentParser()
argparser.add_argument('--pretrained_model', type=str, required=True, help='Path to the pretrained model directory.')
argparser.add_argument('--input_dir', type=str, required=True, help='Path to the input audio file.')
argparser.add_argument('--output_dir', type=str, required=True, help='Directory to save the generated audio files.')
argparser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
argparser.add_argument('--copy_src_to_output_dir', action='store_true', help='Whether to copy the source audio to the output directory.')
args = argparser.parse_args()

seed_everything(args.seed)

device = 0
model_id = args.pretrained_model
attn_implementation = 'sdpa'

model = TasteForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation)

model = model.to(device)
model.eval()

processor = TasteProcessor.from_pretrained(model_id)
generator = processor.get_generator(device=device)

generate_kwargs = dict(
    llm_tokenizer=processor.llm_tokenizer,
    asr_tokenizer=processor.audio_tokenizer,
    extra_words=16,
    text_top_p=0.3,
    taste_top_p=0.0, # not activated for audio embedding continuation
    text_temperature=0.5,
    repetition_penalty=1.1,
    debug=True,
)
# find all audio files in the input directory
conditional_audio_paths = list(glob.glob(os.path.join(args.input_dir, '*.*')))
sampling_rate = 16000

data = [
    processor(
        audio_path,
        sampling_rate,
        ref_audio_list=[audio_path]
    )
    for audio_path in conditional_audio_paths
]

for audio_path in tqdm(conditional_audio_paths):
    audio_fname = os.path.basename(audio_path).split('.')
    audio_fid = audio_fname[0]
    _data = processor(
        audio_path,
        sampling_rate,
        ref_audio_list=[audio_path],
        output_text_info=True
    )
    inputs = _data.to(device=device)
    print(inputs.keys())
    output = model.inference_completion(
        **inputs,
        conditional_mode='audio',
        **generate_kwargs,
    )
    tts_speech, tts_sr = generator.inference(
        speech_token_ids=output['speech_token_ids'], 
        speech_token_lengths=output['speech_token_lengths'],
        flow_embedding=inputs['speaker_embeds']
    )
    if args.copy_src_to_output_dir:
        src_output_fpath = os.path.join(args.output_dir, f"{'.'.join(audio_fname)}")
        copyfile(audio_path, src_output_fpath)
    contd_text = _data['text'][0]
    gened_text = output['generated_text']
    output_text_fpath = os.path.join(args.output_dir, f'{audio_fid}_contd.txt')
    with open(output_text_fpath, 'w') as f:
        f.write(f'=== Conditional Text ===\n{contd_text}\n\n')
        f.write(f'=== Generated Continuation Text ===\n{gened_text}\n')
    output_fpath = os.path.join(args.output_dir, f'{audio_fid}_contd.wav')
    torchaudio.save(output_fpath, tts_speech, tts_sr)