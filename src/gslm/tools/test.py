import json

# Load the original list-format JSON
with open("/home/ubuntu/speech_ppl/src/mfa/phone_extraction.json", "r") as f:
    data = json.load(f)

# Re-index by audio_id (or filename if you prefer, e.g. f"{entry['audio_id']}.wav")
indexed = {entry["audio_id"]: entry for entry in data}

# Save the result
with open("/home/ubuntu/speech_ppl/src/metrics/alignments.json", "w") as f:
    json.dump(indexed, f, indent=2)

print(f"Indexed {len(indexed)} entries.")