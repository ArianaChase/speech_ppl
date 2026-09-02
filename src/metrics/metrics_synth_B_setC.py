import glob, os, pandas as pd, numpy as np
D = "work/outputs/cosyvoice"   # <-- folder holding the six setC per-token CSVs
def tot(audio, text):
    f = glob.glob(f"{D}/*_{audio}_{text}_setC_per_token_losses.csv")[0]
    d = pd.read_csv(f, dtype={'filename': str})
    return d.groupby('filename').ppl_loss.sum()          # whole utterance, same region for both texts
S = {(a,t): tot(a,t) for a in ('real','ref') for t in ('real','foil','ref')}
ids = sorted(set.intersection(*[set(v.index) for v in S.values()]))
g = lambda a,t: S[(a,t)].loc[ids].values
def pct(x, y): return 100*np.mean(x < y)                  # lower loss = better fit


print("n =", len(ids))
print("2AFC  actual beats foil (real audio)   :", round(pct(g('real','real'), g('real','foil')), 2))
print("      actual beats canonical           :", round(pct(g('real','real'), g('real','ref')), 2))
print("SANITY canonical beats actual (ref aud):", round(pct(g('ref','ref'), g('ref','real')), 2))
print("      canonical beats foil             :", round(pct(g('ref','ref'), g('ref','foil')), 2))
print("TEXT-PRIOR CONTROL actual beats foil on ref audio:", round(pct(g('ref','real'), g('ref','foil')), 2))

d_real = g('real','foil') - g('real','real')   # how much better the true text fits the LEARNER audio
d_ref  = g('ref','foil')  - g('ref','real')    # the same preference on error-free audio
print("reference-corrected 2AFC:", round(100*np.mean(d_real > d_ref), 2))