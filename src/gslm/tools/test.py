import json

PATH = "/home/u5504709/new_work/speech_ppl/speechocean762/resource/scores.json"

with open(PATH, "r") as f:
    scores = json.load(f)

phone_1s = 0
phone_0s = 0
phone_count = 0
word_1s = 0
word_0s = 0
word_count = 0
utt_1s = 0
utt_0s = 0
utt_count = 0

for filename, item in scores.items():
    utt_count += 1
    if item['accuracy'] > 3:
        utt_1s += 1
    else:
        utt_0s += 1

    for word in item['words']:
        word_count += 1
        if word['accuracy'] > 3:
            word_1s += 1
        else:
            word_0s += 1

        for phone in word['phones-accuracy']:
            phone_count += 1
            if phone > 0.5:
                phone_1s += 1
            else:
                phone_0s += 1

print(f"{phone_1s / phone_count * 100}% of phones are correct, {phone_0s/phone_count * 100}% are incorrect.")
print(f"{word_1s / word_count * 100}% of phones are correct, {word_0s/word_count * 100}% are incorrect.")
print(f"{utt_1s / utt_count * 100}% of phones are correct, {utt_0s/utt_count * 100}% are incorrect.")

        
    



