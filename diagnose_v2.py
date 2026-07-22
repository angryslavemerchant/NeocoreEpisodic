"""Decomposition diagnostics for the v2 ignition failures ($0, local).
Three gates failed on incremental guesses; this measures the links
directly instead of guessing again.

A. KEY CONFUSABILITY: for each pooling scheme, cosine similarity of
   book keys for fact pairs that SHARE an entity vs random pairs.
   The read circuit can only be learned if same-entity facts are
   separable (a subject's query must pick founded vs works_at vs
   makes for the same company).
B. PAYLOAD DECODABILITY: with tied embeddings, the head applied to a
   fact's payload should rank the ANSWER token highly. Reports mean
   rank/top-5 rate of the answer token in emb @ payload, per scheme.
C. V1 REFERENCE: same measurements on v1-style short templates
   (the world that ignited) for calibration.

Schemes: flat mean | idf | idf^0.5 | idf-capped (clip extremes).
"""

import math
import random

import torch
import torch.nn.functional as F

import stream_text_v2 as W

torch.manual_seed(0)
W.load_bank()
nv = W.build_vocab()
W.UNKNOWN_IDS = W.enc_c(" unknown")
W.build_idf()
d = 256
emb = F.normalize(torch.randn(nv, d), dim=-1) * 0.5  # random tied emb

IDF = W._IDF.clone()
SCHEMES = {
    "flat": torch.ones(nv),
    "idf": IDF,
    "idf_sqrt": IDF.clamp(min=1e-3).sqrt(),
    "idf_cap": IDF.clamp(min=0.5, max=2.5),
}


def pool(ids, wvec):
    t = torch.tensor(ids)
    w = wvec[t].unsqueeze(-1)
    return (emb[t] * w).sum(0) / w.sum().clamp(min=1e-6)


def analyze(lifetimes, label, texts_of):
    print(f"\n=== {label}")
    for name, wvec in SCHEMES.items():
        same_e, rand_p, ranks, top5 = [], [], [], []
        for lt, facts, answers, ent_of in lifetimes:
            keys = []
            for texts in facts:
                ks = [pool(W.enc_c(t), wvec) for t in texts]
                keys.append(F.normalize(
                    torch.stack(ks).mean(0), dim=-1))
            keys = torch.stack(keys)
            n = len(facts)
            for i in range(n):
                for j in range(i + 1, n):
                    c = float(keys[i] @ keys[j])
                    if ent_of[i] & ent_of[j]:
                        same_e.append(c)
                    elif random.random() < 0.05:
                        rand_p.append(c)
            # payload decodability: answer-token rank in emb @ payload
            for i, ans_ids in enumerate(answers):
                if not ans_ids:
                    continue
                pay = torch.stack(
                    [pool(W.enc_c(t), wvec) for t in facts[i]]).mean(0)
                logits = emb @ pay
                r = int((logits > logits[ans_ids[0]]).sum())
                ranks.append(r)
                top5.append(r < 5)
        print(f"  {name:9s} same-entity cos {sum(same_e)/len(same_e):.3f}"
              f"  random cos {sum(rand_p)/max(len(rand_p),1):.3f}"
              f"  ans-rank median {sorted(ranks)[len(ranks)//2]:4d}"
              f"  ans-top5 {100*sum(top5)/len(top5):.0f}%")


# --- v2 lifetimes -----------------------------------------------------
rng = random.Random(1)
v2_lts = []
for _ in range(4):
    lt = W.Lifetime(rng, stmts=3)
    facts, answers, ent_of = [], [], []
    for key, fid in lt.fid.items():
        pass
    # rebuild per-fact info: reconstruct from lt.fid keys
    fid_items = sorted(lt.fid.items(), key=lambda kv: kv[1])
    fact_texts = [[] for _ in range(lt.n_facts)]
    for (ids, kind, fid, *_r) in lt.docs:
        if kind == "fact":
            fact_texts[fid].append(W.ENC.decode(
                [k for g, k in W._REMAP.items() if False]) if False
                else None)
    # simpler: regenerate texts by re-rendering — use doc ids directly
    fact_ids = [[] for _ in range(lt.n_facts)]
    for (ids, kind, fid, *_r) in lt.docs:
        if kind == "fact":
            fact_ids[fid].append(ids)
    facts2, answers2, ent2 = [], [], []
    for (key, fid) in fid_items:
        rel = key[0]
        ent = key[1]
        ents = (set(ent) if isinstance(ent, frozenset)
                else {ent})
        # answer ids per relation
        if rel == "founded":
            ans = W.enc_c(" " + lt.founder[ent])
        elif rel == "industry":
            ans = W.enc_c(" " + lt.industry[ent])
        elif rel == "based_in":
            ans = W.enc_c(" " + lt.city[ent])
        elif rel == "makes":
            ans = W.enc_c(" " + lt.makes[ent])
        elif rel == "works_as":
            ans = W.enc_c(" " + lt.job[ent])
        elif rel == "lives_in":
            ans = W.enc_c(" " + lt.home[ent])
        elif rel == "works_at":
            ans = W.enc_c(" " + lt.employer[ent])
            ents.add(lt.employer[ent])
        else:
            a, b = tuple(ent)
            ans = W.enc_c(" " + b)
        facts2.append(fact_ids[fid])
        answers2.append(ans)
        ent2.append(ents)
    # NOTE: facts2 holds token-id lists already; pool() re-encodes
    v2_lts.append((lt, facts2, answers2, ent2))


def analyze_ids(lifetimes, label):
    print(f"\n=== {label}")
    for name, wvec in SCHEMES.items():
        same_e, rand_p, ranks, top5 = [], [], [], []
        for lt, facts, answers, ent_of in lifetimes:
            keys = []
            for id_lists in facts:
                ks = [pool_ids(ids, wvec) for ids in id_lists]
                keys.append(F.normalize(
                    torch.stack(ks).mean(0), dim=-1))
            keys = torch.stack(keys)
            n = len(facts)
            for i in range(n):
                for j in range(i + 1, n):
                    c = float(keys[i] @ keys[j])
                    if ent_of[i] & ent_of[j]:
                        same_e.append(c)
                    elif random.random() < 0.05:
                        rand_p.append(c)
            for i, ans_ids in enumerate(answers):
                if not ans_ids:
                    continue
                pay = torch.stack([pool_ids(ids, wvec)
                                   for ids in facts[i]]).mean(0)
                logits = emb @ pay
                r = int((logits > logits[ans_ids[0]]).sum())
                ranks.append(r)
                top5.append(r < 5)
        print(f"  {name:9s} same-entity cos "
              f"{sum(same_e)/len(same_e):.3f}"
              f"  random cos {sum(rand_p)/max(len(rand_p),1):.3f}"
              f"  ans-rank median {sorted(ranks)[len(ranks)//2]:4d}"
              f"  ans-top5 {100*sum(top5)/len(top5):.0f}%")


def pool_ids(ids, wvec):
    t = torch.tensor(ids)
    w = wvec[t].unsqueeze(-1)
    return (emb[t] * w).sum(0) / w.sum().clamp(min=1e-6)


analyze_ids(v2_lts, "V2 bank world (3 stmts, 44-template bank)")

# --- v1-style reference: short fixed templates ------------------------
V1_T = {
    "founded": ["{P} founded {C}."],
    "industry": ["{C} is a {I} company."],
    "based_in": ["{C} is headquartered in {X}."],
    "makes": ["{C} makes {PROD}."],
    "works_as": ["{P} works as a {J}."],
    "lives_in": ["{P} lives in {X}."],
    "works_at": ["{P} works at {C}."],
    "partner": ["{P} is married to {P2}."],
}
v1_lts = []
for _ in range(4):
    lt = W.Lifetime(rng, stmts=3)
    fid_items = sorted(lt.fid.items(), key=lambda kv: kv[1])
    facts2, answers2, ent2 = [], [], []
    for (key, fid) in fid_items:
        rel, ent = key[0], key[1]
        ents = set(ent) if isinstance(ent, frozenset) else {ent}
        if rel == "founded":
            txt = V1_T[rel][0].format(P=lt.founder[ent], C=ent)
            ans = W.enc_c(" " + lt.founder[ent])
        elif rel == "industry":
            txt = V1_T[rel][0].format(C=ent, I=lt.industry[ent])
            ans = W.enc_c(" " + lt.industry[ent])
        elif rel == "based_in":
            txt = V1_T[rel][0].format(C=ent, X=lt.city[ent])
            ans = W.enc_c(" " + lt.city[ent])
        elif rel == "makes":
            txt = V1_T[rel][0].format(C=ent, PROD=lt.makes[ent])
            ans = W.enc_c(" " + lt.makes[ent])
        elif rel == "works_as":
            txt = V1_T[rel][0].format(P=ent, J=lt.job[ent])
            ans = W.enc_c(" " + lt.job[ent])
        elif rel == "lives_in":
            txt = V1_T[rel][0].format(P=ent, X=lt.home[ent])
            ans = W.enc_c(" " + lt.home[ent])
        elif rel == "works_at":
            txt = V1_T[rel][0].format(P=ent, C=lt.employer[ent])
            ans = W.enc_c(" " + lt.employer[ent])
            ents.add(lt.employer[ent])
        else:
            a, b = tuple(ent)
            txt = V1_T[rel][0].format(P=a, P2=b)
            ans = W.enc_c(" " + b)
        facts2.append([W.enc_c(txt)] * 1)
        answers2.append(ans)
        ent2.append(ents)
    v1_lts.append((lt, facts2, answers2, ent2))

analyze_ids(v1_lts, "V1-style short fixed templates (reference)")
