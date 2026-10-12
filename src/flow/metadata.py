import json

# Load speaker -> age mapping
spk2age = {}

with open("/home/u5504709/new_work/speech_ppl/speechocean762/train/spk2gender", "r") as f:
    for line in f:
        spk, gender = line.strip().split()
        spk2age[spk] = gender

with open("/home/u5504709/new_work/speech_ppl/speechocean762/test/spk2gender", "r") as f:
    for line in f:
        spk, age = line.strip().split()
        spk2age[spk] = gender

# Load JSON
with open("/home/u5504709/new_work/speech_ppl/src/output.json", "r") as f:
    data = json.load(f)

# data is a dict
for file in data:
    spk = file[1:5]
    if spk in spk2age:
        print(spk2age[spk])
        data[file]["gender"] = spk2age[spk]
    else:
        data[file]["gender"] = None

# Save
with open("output.json", "w") as f:
    json.dump(data, f, indent=4)