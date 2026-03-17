import os
import json

train_text = "/home/u5504709/new_work/speech_ppl/speechocean762/train/text"
test_text = "/home/u5504709/new_work/speech_ppl/speechocean762/test/text"

    
def create_lab_files(file):
    root_path = "/home/u5504709/new_work/speech_ppl/src/mfa/WAVE"

    with open(file, "r") as f:
        #counter = 0
        for line in f:
            #if (counter > 5):
            #    break
            cleaned = line.strip()
            speaker_id = "SPEAKER" + cleaned[1:5]
            audio_id = cleaned[0:9]
            print(f"Speaker ID: {speaker_id}, Audio ID: {audio_id}")

            transcription = cleaned[10:]
            print(transcription)

            with open(os.path.join(root_path, speaker_id, audio_id + ".lab"), "w") as lab_file:
                lab_file.write(transcription)

            #counter += 1
                

create_lab_files(train_text)
create_lab_files(test_text)


