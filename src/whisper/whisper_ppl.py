import torch
import torch.nn.functional as F
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

WAV        = "/home/ubuntu/speech_ppl/speechocean762/WAVE/SPEAKER0001/000010011.WAV"
CANDIDATE  = "we call it bear"          # the transcript you're scoring

processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
model     = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------- audio -> encoder hidden states (run once, reuse per candidate) ----------
wav, _ = librosa.load(WAV, sr=16000, mono=True)
features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device)

with torch.no_grad():
    hidden = model.model.encoder(features).last_hidden_state

# ---------- candidate text -> decoder input ----------
tokenizer = processor.tokenizer
prefix = tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", "<|notimestamps|>"])
text_ids = tokenizer.encode(" " + CANDIDATE, add_special_tokens=False)

dec = torch.tensor([prefix + text_ids + [tokenizer.eos_token_id]], device=device)
# dec looks like: [start, notimestamps, tok1, tok2, ..., tokN, eos]

# ---------- teacher-forced forward pass ----------
with torch.no_grad():
    out = model(encoder_outputs=(hidden,), decoder_input_ids=dec[:, :-1])
    # dec[:, :-1] drops eos from the input — the model should never see the answer
    # it's about to be asked to predict

logp = torch.log_softmax(out.logits[0].float(), dim=-1)

# ---------- surprisal ----------
# position p in dec[:, :-1] predicts dec[p+1]. The prefix has 2 tokens (indices 0,1),
# so the first real text token is predicted by position 1, and text runs to the end.
targets = dec[0, 2:]                      # every real text token, plus eos at the end
print(tokenizer.convert_ids_to_tokens(dec[0].tolist()))
preds   = logp[1:1 + len(targets)]        # the positions that predict them

per_token = -preds[torch.arange(len(targets)), targets]
sentence_surprisal = per_token.sum().item()

print("per-token surprisal:", per_token.tolist())
print("sentence surprisal (sum, nats):", sentence_surprisal)