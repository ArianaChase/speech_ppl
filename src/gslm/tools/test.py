import json
from tqdm import tqdm
import os
from datasets import Dataset, Features, Value, Audio, Sequence

def process_speechocean(input_dataset, alignments_path, labels_dict):

        audio_file_info = []
        spk_count = 0

        pbar = tqdm(sorted(os.listdir(input_dataset)))

        with open(alignments_path, 'r') as f:
            alignment_list = json.load(f)

        for spk_dir in pbar:
            spk_count += 1
            speaker = spk_dir[7:None]
            pbar.set_description(f"Processing speaker: {speaker}")
            spk_dir_path = os.path.join(input_dataset, spk_dir)
            for audio_file in os.listdir(spk_dir_path):
                audio_path = os.path.join(spk_dir_path, audio_file)
                filename = os.path.basename(audio_path)[0:9]
                alignment_obj = next((item for item in alignment_list if item.get('audio_id') == filename), None)
                human_annotation_obj = labels_dict.get(filename)

                if alignment_obj != None:

                    audio_file_info.append({
                        "speaker" : speaker,
                        "filename" : filename,
                        "path" : audio_path,
                        "text" : human_annotation_obj['text'],
                        "human_annotations" : json.dumps(human_annotation_obj, ensure_ascii=False),
                        "word_alignments" : json.dumps(alignment_obj['word_alignment'], ensure_ascii=False),
                        "phone_alignments" : json.dumps(alignment_obj['phone_alignment'], ensure_ascii=False)
                    })
                          
        return audio_file_info

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

SPEECHOCEAN_PATH = "/home/u5504709/new_work/speech_ppl/speechocean762/WAVE"
ALIGNMENTS = "/home/u5504709/new_work/speech_ppl/src/mfa/phone_extraction.json"
LABELS = '/home/u5504709/new_work/speech_ppl/speechocean762/resource/scores.json'

human_scores = parse_human_annotations(LABELS)
processed_dataset = process_speechocean(SPEECHOCEAN_PATH, ALIGNMENTS, human_scores)

with open('/home/u5504709/new_work/speech_ppl/src/gslm/tools/speechocean_dataset.json', 'w') as f:
    json.dump(processed_dataset, f)


features = Features({
    "speaker": Value("string"),
    "filename": Value("string"),
    "path": Audio(sampling_rate=16000),
    "text": Value("string"),

    "human_annotations": Value("string"),
    "word_alignments": Value("string"),
    "phone_alignments": Value("string"),
})

dataset = Dataset.from_list(
    processed_dataset,
    features=features,
)

dataset.push_to_hub('peggy2009/speechocean_with_mfa')

print(f"Length: {len(processed_dataset)}")