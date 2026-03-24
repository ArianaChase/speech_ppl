import os
from pathlib import Path


dataset = "/home/u5504709/new_work/speech_ppl/speechocean762/WAVE"

def create_predict_id(dataset):
    with open("/home/u5504709/new_work/speech_ppl/src/flow-slm/predict_id.txt", "w") as f:
        total = 0
        for speaker in os.listdir(dataset):
            print("spekaer: ", speaker[7:11])

            if speaker[7:11] != "1076":
                file_count = 0

                for file in os.listdir(os.path.join(dataset, speaker)):
                    if file_count >= 20:
                        break
                    print("file: ", file)
                    f.write("%s/%s\n" % (speaker, Path(file).stem))
                    total += 1
                    file_count += 1
    
    print("Total files processed: ", total)

create_predict_id(dataset)

                
