import torch
import torchaudio
print(torch.__version__)
print(torch.version.cuda)      # should print something like '12.1'
print(torchaudio.version.cuda)      # should print something like '12.1'

print(torch.backends.cudnn.is_available())  # should be True