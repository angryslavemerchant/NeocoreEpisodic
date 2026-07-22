"""Validate + assemble the template bank (stage 2 of the real-text
campaign). Generation is done by dispatched subagents (haiku-class)
writing JSON shards; NOTHING enters the assembled bank unvalidated —
propose-verify-file applied to dataset construction.

Checks per template:
- required slots present EXACTLY once, no unknown slots
- renders under MAX_BPE tokens with worst-case fillers
- ends correctly (statements: '.', questions: 'A:')
- no gendered pronouns (nonce persons are genderless)
- no real-looking proper nouns outside slots (heuristic: capitalized
  words not at sentence start and not slot/closed-set words)
- dedup by lowercased alnum-normalized form (within and across shards)

Usage: validate_bank.py shard1.json shard2.json ... -o templates_bank.json
Prints per-category accept/reject counts and shortfalls vs QUOTA.
"""

import argparse
import json
import re
import sys

import tiktoken

ENC = tiktoken.get_encoding("gpt2")
MAX_BPE = 26

SLOTS = {
    "founded": ["P", "C"], "industry": ["C", "I"], "based_in": ["C", "X"],
    "works_at": ["P", "C"], "works_as": ["P", "J"], "partner": ["P", "P2"],
    "makes": ["C", "PROD"], "lives_in": ["P", "X"],
}
Q_SLOTS = {
    "q_founded": ["C"], "q_industry": ["C"], "q_based_in": ["C"],
    "q_works_at": ["P"], "q_works_as": ["P"], "q_partner": ["P"],
    "q_makes": ["C"], "q_lives_in": ["P"],
    "q_hop_industry": ["P"], "q_hop_city": ["P"], "q_hop_job": ["P"],
    "q_hop_lives": ["P"], "q_hop_works_industry": ["P"],
}
QUOTA = {**{k: 35 for k in SLOTS}, **{k: 12 for k in Q_SLOTS},
         "fillers": 100}
TEST_FILL = {"P": "Aaaa Bbbb", "P2": "Cccc Dddd", "C": "Eeee Ffff Gggg",
             "I": "publishing", "X": "Quebec", "J": "carpenter",
             "PROD": "Hhhh Iiii"}
GENDERED = re.compile(r"\b(he|she|his|her|him|hers|himself|herself)\b",
                      re.IGNORECASE)
ALLOWED_CAPS = {"Q", "A"}  # question prefix tokens

# editorial blocklist (normalized): phrasings whose meaning drifts —
# e.g. "founded in <city>" reads as founding-location, not headquarters
BLOCK = {
    "qthecitywherepfoundedthecompanyis?a:",
    "qpfoundedacompanyinwhatcity?a:",
    "qpcreatedacompanyinwhatcity?a:",
    "qpstartedacompanyinwhichcity?a:",
    "qinwhatcityisthecompanypfounded?a:",
    "qinwhichcityisthecompanypfounded?a:",
    # statements that assert no extractable fact
    "pandcareaformidablepair",
    "cstandsasamonumenttopsdrive",
    "cbelongstox",
}


def norm(t):
    return re.sub(r"[^a-z0-9{}]", "", t.lower())


def check(cat, text):
    slots = SLOTS.get(cat) or Q_SLOTS.get(cat) or []
    for s in slots:
        if text.count("{" + s + "}") != 1:
            return f"slot {{{s}}} not exactly once"
    found = set(re.findall(r"\{(\w+)\}", text))
    if found - set(slots):
        return f"unknown slots {found - set(slots)}"
    is_q = cat.startswith("q_")
    if is_q and not text.rstrip().endswith("A:"):
        return "question must end with 'A:'"
    if not is_q and not text.rstrip().endswith("."):
        return "statement must end with '.'"
    if GENDERED.search(re.sub(r"\{\w+\}", "", text)):
        return "gendered pronoun"
    rendered = text
    for k, v in TEST_FILL.items():
        rendered = rendered.replace("{" + k + "}", v)
    if len(ENC.encode(rendered)) > MAX_BPE:
        return f"too long ({len(ENC.encode(rendered))} BPE)"
    body = re.sub(r"\{\w+\}", "SLOT", text)
    if is_q:
        # skip the "Q: " prefix and the question's own first word
        m = re.match(r"\s*Q:\s*\w+", body)
        scan = body[m.end():] if m else body
    else:
        scan = body[1:]
    for w in re.findall(r"\b[A-Z][a-z]+\b", scan):
        if w not in ALLOWED_CAPS and w != "SLOT":
            return f"stray proper noun '{w}'"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("-o", "--out", default="templates_bank.json")
    args = ap.parse_args()
    bank = {k: [] for k in list(SLOTS) + list(Q_SLOTS) + ["fillers"]}
    seen = set()
    rejects = {}
    for path in args.shards:
        with open(path, encoding="utf-8") as f:
            shard = json.load(f)
        for cat, items in shard.items():
            if cat not in bank:
                rejects.setdefault("unknown-category", []).append(cat)
                continue
            for t in items:
                if not isinstance(t, str):
                    continue
                # normalize doubled braces ({{P}} -> {P}) before checks
                t = re.sub(r"\{\{(\w+)\}\}", r"{\1}", t.strip())
                if norm(t).replace("{", "").replace("}", "") in BLOCK:
                    rejects.setdefault(f"{cat}: editorial block",
                                       []).append(t)
                    continue
                if cat == "fillers":
                    err = ("has slots" if "{" in t else
                           None if t.endswith(".") else "no period")
                    if err is None and len(ENC.encode(t)) > MAX_BPE:
                        err = "too long"
                else:
                    err = check(cat, t)
                key = (cat, norm(t))
                if err is None and key in seen:
                    err = "duplicate"
                if err is None:
                    seen.add(key)
                    bank[cat].append(t)
                else:
                    rejects.setdefault(f"{cat}: {err}", []).append(t)
    print("=== accepted per category (vs quota):")
    short = {}
    for cat, quota in QUOTA.items():
        n = len(bank[cat])
        flag = "" if n >= quota else "  << SHORT"
        if n < quota:
            short[cat] = quota - n
        print(f"  {cat:22s} {n:3d}/{quota}{flag}")
    print(f"\n=== rejects ({sum(len(v) for v in rejects.values())}):")
    for reason, items in sorted(rejects.items()):
        print(f"  {len(items):3d}  {reason}   e.g. {items[0][:60]!r}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=1, ensure_ascii=False)
    print(f"\nbank -> {args.out}")
    if short:
        print(f"SHORTFALLS: {short}")
        sys.exit(2)


if __name__ == "__main__":
    main()
