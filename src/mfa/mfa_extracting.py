import os
import json
from praatio import textgrid

tg_directory = "/home/u5504709/new_work/speech_ppl/work/outputs/textgrids"
output_json = "/home/u5504709/new_work/speech_ppl/src/mfa/phone_extraction.json"

def textgrid_to_json(file, tier_name):
    tg = textgrid.openTextgrid(file, False)
    tier = tg.getTier(tier_name)
    alignment_list =[]
    
    for entry in tier.entries:
        if hasattr(entry, 'start'):
            alignment_list.append({
                "start": round(entry.start, 3),
                "end": round(entry.end, 3),
                "label": entry.label
            })
        elif hasattr(entry, 'time'):
            alignment_list.append({
                "time": round(entry.time, 3),
                "label": entry.label
            })
    return alignment_list

dataset =[]

for speaker_dir in os.listdir(tg_directory):
    for filename in os.listdir(os.path.join(tg_directory, speaker_dir)):
        if filename.endswith(".TextGrid"):
            full_path = os.path.join(tg_directory, speaker_dir, filename)
            
            word_alignment_list = textgrid_to_json(full_path, "words")
            phone_alignment_list = textgrid_to_json(full_path, "phones")

            # filename is already the base name, no need for os.path.basename
            speaker = speaker_dir[7:11]
            audio_id = filename[0:9]

            file_dict = {
                "speaker": speaker,
                "audio_id": audio_id,
                "word_alignment": word_alignment_list,
                "phone_alignment": phone_alignment_list
            }
            dataset.append(file_dict)

print(dataset)

# Write the entire dataset as a single JSON array
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"Saved {len(dataset)} entries to {output_json}")