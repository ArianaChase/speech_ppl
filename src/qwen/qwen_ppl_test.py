import torch, librosa, numpy as np
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts import Qwen3TTSTokenizer
import torch.nn.functional as F

WAV        = "/home/ubuntu/speech_ppl/speechocean762/WAVE/SPEAKER0001/000010011.WAV"      # the audio being scored
REF_WAV    = WAV      # any clip of a voice, for the speaker slot
TRANSCRIPT = "WE CALL IT BEAR"

# ---------- load ----------
tts   = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", torch_dtype=torch.bfloat16)
codec = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")

model, cfg = tts.model, tts.model.config
tk  = cfg.talker_config
dev = model.device

# ---------- inputs ----------
# audio -> one row per 83 ms, 16 columns. These are the "right answers".
codes = codec.encode([WAV]).audio_codes[0].long()
C = len(codes)

# text -> ids. The processor adds a trailing block the model doesn't want, so drop 5.
prompt = f"<|im_start|>assistant\n{TRANSCRIPT}<|im_end|>\n<|im_start|>assistant\n"
tid = tts.processor(text=prompt, return_tensors="pt")["input_ids"][0][:-5]
L = len(tid)

# ---------- build the two channels ----------
# The sequence is: 8 control slots, then the text, then one slot per audio frame.
A = 8 + L - 1        # index where the first audio frame sits
N = A + C + 1        # total length (last slot is end-of-speech)

txt = torch.full((N,), cfg.tts_pad_token_id)   # text channel
cdc = torch.zeros(N, dtype=torch.long)         # codec channel

txt[0:3]     = tid[:3]
txt[7]       = cfg.tts_bos_token_id
txt[8:A-2]   = tid[3:]
txt[A-2]     = cfg.tts_eos_token_id

cdc[3:8]     = torch.tensor([tk.codec_nothink_id, tk.codec_think_bos_id,
                             tk.codec_think_eos_id, 0, tk.codec_pad_id])
cdc[8:A-1]   = tk.codec_pad_id
cdc[A-1]     = tk.codec_bos_id
cdc[A:A+C]   = codes[:, 0]
cdc[A+C]     = tk.codec_eos_token_id

# ---------- turn ids into one input array ----------
emb = model.talker.model.text_embedding(txt.to(dev))

c = model.talker.model.codec_embedding(cdc.to(dev))
c[:3] = 0                                              # nothing in the codec channel yet
c[6]  = model.extract_speaker_embedding(               # slot 6 is reserved for the voice
    librosa.load(REF_WAV, sr=24000, mono=True)[0].astype(np.float32), 24000)[0]
emb = emb + c

# feed the true fine codebooks at each frame — this is the teacher forcing
for i in range(1, 16):
    emb[A:A+C] += model.talker.code_predictor.get_input_embeddings()[i-1](codes[:, i].to(dev))

# ---------- one forward pass ----------
with torch.no_grad():
    out = model.talker(inputs_embeds=emb[None].to(model.dtype),
                       attention_mask=torch.ones(1, N, dtype=torch.long, device=dev),
                       use_cache=False)

# ---------- surprisal ----------
pos  = torch.arange(A, A + C + 1)              # every audio frame, plus end-of-speech
per_token_loss = F.cross_entropy(
    out.logits[0, pos - 1],      # (C+1, 2048) — one row of logits per audio slot
    cdc[pos].to(dev),            # (C+1,) — the true catalog number at each slot
    reduction='none'
)
print(per_token_loss)