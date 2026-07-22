"""SCAN-proper under the frozen-kernel discipline — brittleness bout #3.

Real dataset (Lake & Baroni 2018, github.com/brendenlake/SCAN), not a
toy world. Zero trained parameters, zero gradients: word meanings are
acquired by EXACT HYPOTHESIS ELIMINATION over demonstrations, filed in
a book (word -> (role, program)), executed by a frozen kernel.

What is authored (kernel invariants, computation-shaped only):
- Clause template order: VERB [STRUCT] [DIR] [COUNT]; a command is one
  clause or two joined by a CONNECTOR (English surface order — same
  status as toy v5's frames).
- Program spaces per role (no SCAN word is named anywhere):
    VERB   -> an action sequence (possibly empty)
    DIR    -> a single action atom
    STRUCT -> program over (dir d, verb v): family 'pre' = d^k v,
              family 'inter' = (d v)^k, k in 1..4   (8 candidates)
    COUNT  -> repeat whole clause body n in {2,3,4}
    CONN   -> order-blind concat + swap bit in {0,1}
  Plain "V D" is struct ('pre',1) — the default program.
What is learned (from demos, exactly): every word's role AND its
program parameters, by intersecting the consistent-hypothesis sets
across demonstrations. A word's hypothesis dies the moment one demo
admits no consistent parse using it. Nothing is soft; contradictions
are reported, not smoothed.

PRE-REGISTERED (before first run on test files):
1. Oracle-role executor (test-only scaffold) matches 100% of train
   demos — else the grammar schema itself is wrong and we report that.
2. The learner grounds all 13 words with zero contradictions from the
   short-command prefix of training data.
3. Test exact match: simple 100.0, addprim_jump 100.0 (add-jump at
   full strength — 'jump' trains ONLY in isolation), length 100.0
   (train outputs <=22 actions, test 24-48).
4. NO new operation may be admitted to make anything fit (op set:
   emit, repeat-power, interleave, concat-swap). A schema gap = the
   finding, not a patch site.
Literature contrast (published, not rerun): seq2seq ~99.8 simple,
~1.2 addprim_jump, ~13.8 length (Lake & Baroni 2018).
"""

import argparse
import time
from collections import defaultdict
from itertools import product

ROLES = ("V", "D", "S", "C", "N")
TEMPLATES = [("V",), ("V", "D"), ("V", "S", "D"), ("V", "C"),
             ("V", "D", "C"), ("V", "S", "D", "C")]
STRUCT_SPACE = [(fam, k) for fam in ("pre", "inter") for k in (1, 2, 3, 4)]
COUNT_SPACE = [2, 3, 4]
CONN_SPACE = [0, 1]


def load(path):
    demos = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inp, out = line.split(" OUT: ")
            words = tuple(inp.replace("IN: ", "").split())
            acts = tuple(out.split())
            demos.append((words, acts))
    return demos


# ---------------------------------------------------------------- kernel
def exec_clause(verb_em, dir_atom=None, struct=("pre", 1), count=1):
    """Frozen executor. verb_em: tuple of atoms (may be empty)."""
    if dir_atom is None:
        body = tuple(verb_em)
    else:
        fam, k = struct
        if fam == "pre":
            body = (dir_atom,) * k + tuple(verb_em)
        else:
            body = ((dir_atom,) + tuple(verb_em)) * k
    return body * count


def exec_command(clauses, swap=0):
    if len(clauses) == 1:
        return clauses[0]
    a, b = clauses
    return b + a if swap else a + b


# ---------------------------------------------------------------- book
class Book:
    """word -> candidate (role, param) hypotheses; exact elimination."""

    def __init__(self):
        self.roles = defaultdict(lambda: set(ROLES))
        self.verb_em = {}      # word -> set of emission tuples (None=free)
        self.dir_atom = {}     # word -> set of atoms (None=free)
        self.struct = defaultdict(lambda: set(STRUCT_SPACE))
        self.count = defaultdict(lambda: set(COUNT_SPACE))
        self.conn = defaultdict(lambda: set(CONN_SPACE))
        self.contradictions = []

    def grounded(self, w):
        """Fully resolved: one role, and its param set is a singleton."""
        if len(self.roles[w]) != 1:
            return False
        r = next(iter(self.roles[w]))
        return len(self._params(w, r) or [1]) == 1 and \
            self._params(w, r) is not None

    def _params(self, w, r):
        if r == "V":
            return self.verb_em.get(w)
        if r == "D":
            return self.dir_atom.get(w)
        if r == "S":
            return self.struct[w]
        if r == "C":
            return self.count[w]
        if r == "N":
            return self.conn[w]

    def param_candidates(self, w, r):
        """Concrete candidate list for enumeration; None = unconstrained."""
        p = self._params(w, r)
        return None if p is None else sorted(p)


# ------------------------------------------------- clause interpretation
def clause_groundings(book, ws, tpl):
    """All consistent full groundings of clause words `ws` under role
    template `tpl`, given current candidate sets. Returns list of
    (output_tuple, bindings) where bindings = {(word, role): param}, or
    the string 'UNK' if >1 unconstrained param blocks enumeration.
    Unconstrained VERB emission is handled by INVERSION downstream (the
    caller solves for it against the demo output), signalled by output
    None with a 'solve' marker in bindings."""
    slots = list(zip(ws, tpl))
    cand_lists, free = [], []
    for w, r in slots:
        c = book.param_candidates(w, r)
        if c is None:
            free.append((w, r))
            cand_lists.append([None])
        else:
            cand_lists.append(c)
    if len(free) > 1:
        return "UNK"
    outs = []
    for combo in product(*cand_lists):
        bind = {}
        verb_em = dir_atom = None
        struct, count = ("pre", 1), 1
        solve_role = None
        for (w, r), p in zip(slots, combo):
            if p is None:
                solve_role = (w, r)
            bind[(w, r)] = p
            if r == "V":
                verb_em = p
            elif r == "D":
                dir_atom = p
            elif r == "S":
                struct = p
            elif r == "C":
                count = p
        if solve_role is None:
            outs.append((exec_clause(verb_em, dir_atom, struct, count),
                         bind))
        else:
            outs.append((None, dict(bind, __solve__=(solve_role, verb_em,
                                                     dir_atom, struct,
                                                     count))))
    return outs


def invert_clause(out, solve_info):
    """Given target clause output and the known parts, solve the one
    free param. Returns param or None if inconsistent."""
    (w, r), verb_em, dir_atom, struct, count = solve_info
    if len(out) % count:
        return None
    blk = out[:len(out) // count]
    if blk * count != out:
        return None
    if r == "V":
        if dir_atom is None:
            return blk
        fam, k = struct
        if fam == "pre":
            if blk[:k] != (dir_atom,) * k:
                return None
            return blk[k:]
        if len(blk) % k:
            return None
        unit = blk[:len(blk) // k]
        if unit * k != blk or not unit or unit[0] != dir_atom:
            return None
        return unit[1:]
    if r == "D":
        if not blk:
            return None
        d = blk[0]
        if exec_clause(verb_em, d, struct, count) != out:
            return None
        return d
    return None


def command_parses(book, words):
    """Enumerate parses: single clause, or clause CONN clause."""
    parses = []
    for tpl in TEMPLATES:
        if len(tpl) == len(words) and all(
                r in book.roles[w] for w, r in zip(words, tpl)):
            parses.append(("one", tpl))
    for i in range(1, len(words) - 1):
        if "N" not in book.roles[words[i]]:
            continue
        for t1 in TEMPLATES:
            if len(t1) != i or not all(
                    r in book.roles[w] for w, r in zip(words[:i], t1)):
                continue
            for t2 in TEMPLATES:
                if len(t2) != len(words) - i - 1 or not all(
                        r in book.roles[w]
                        for w, r in zip(words[i + 1:], t2)):
                    continue
                parses.append(("two", i, t1, t2))
    return parses


def demo_supports(book, words, out):
    """Evaluate every parse; return (conclusive, supports) where
    supports = {(word): set of (role, param)} across consistent
    groundings. conclusive=False if any parse was UNK-blocked (then no
    elimination may be made from this demo)."""
    supports = defaultdict(set)
    conclusive = True
    any_parse = False
    for parse in command_parses(book, words):
        if parse[0] == "one":
            tpl = parse[1]
            gs = clause_groundings(book, words, tpl)
            if gs == "UNK":
                conclusive = False
                continue
            for o, bind in gs:
                if "__solve__" in bind:
                    info = bind.pop("__solve__")
                    p = invert_clause(out, info)
                    if p is None:
                        continue
                    bind[info[0]] = p
                    o = out
                if o == out:
                    any_parse = True
                    for (w, r), p in bind.items():
                        supports[w].add((r, p))
        else:
            _, i, t1, t2 = parse
            ws1, conn, ws2 = words[:i], words[i], words[i + 1:]
            g1 = clause_groundings(book, ws1, t1)
            g2 = clause_groundings(book, ws2, t2)
            if g1 == "UNK" or g2 == "UNK":
                conclusive = False
                continue
            for swap in sorted(book.conn[conn]):
                for o1, b1 in g1:
                    for o2, b2 in g2:
                        s1, s2 = "__solve__" in b1, "__solve__" in b2
                        if s1 and s2:
                            conclusive = False
                            continue
                        bb1, bb2, oo1, oo2 = dict(b1), dict(b2), o1, o2
                        if s1 or s2:
                            # known clause fixes the split point
                            known_o = oo2 if s1 else oo1
                            if known_o is None:
                                conclusive = False
                                continue
                            if s1:
                                seg = (out[:len(out) - len(known_o)]
                                       if swap == 0
                                       else out[len(known_o):])
                                if swap == 0 and out[len(seg):] != known_o:
                                    continue
                                if swap == 1 and out[:len(known_o)] != \
                                        known_o:
                                    continue
                                info = bb1.pop("__solve__")
                                p = invert_clause(seg, info)
                                if p is None:
                                    continue
                                bb1[info[0]] = p
                                oo1 = seg
                            else:
                                seg = (out[len(known_o):] if swap == 0
                                       else out[:len(out) - len(known_o)])
                                if swap == 0 and out[:len(known_o)] != \
                                        known_o:
                                    continue
                                if swap == 1 and out[len(seg):] != \
                                        known_o:
                                    continue
                                info = bb2.pop("__solve__")
                                p = invert_clause(seg, info)
                                if p is None:
                                    continue
                                bb2[info[0]] = p
                                oo2 = seg
                        if exec_command([oo1, oo2], swap) == out:
                            any_parse = True
                            supports[conn].add(("N", swap))
                            for (w, r), p in bb1.items():
                                supports[w].add((r, p))
                            for (w, r), p in bb2.items():
                                supports[w].add((r, p))
    return conclusive and any_parse, supports, any_parse or not conclusive


def learn(book, demos, max_passes=10, verbose=True):
    demos = sorted(demos, key=lambda d: len(d[0]))
    used = 0
    for p in range(max_passes):
        changed = False
        for words, out in demos:
            if all(book.grounded(w) for w in words):
                continue
            conclusive, sup, ok = demo_supports(book, words, out)
            if not ok:
                book.contradictions.append((words, out))
                continue
            if not conclusive:
                continue
            used += 1
            for w in set(words):
                roles_seen = {r for r, _ in sup[w]}
                if book.roles[w] - roles_seen:
                    book.roles[w] &= roles_seen
                    changed = True
                for r in roles_seen:
                    params = {q for rr, q in sup[w] if rr == r}
                    if r == "V":
                        prev = book.verb_em.get(w)
                        new = params if prev is None else prev & params
                        if new != prev:
                            book.verb_em[w] = new
                            changed = True
                    elif r == "D":
                        prev = book.dir_atom.get(w)
                        new = params if prev is None else prev & params
                        if new != prev:
                            book.dir_atom[w] = new
                            changed = True
                    elif r == "S":
                        if book.struct[w] - params:
                            book.struct[w] &= params
                            changed = True
                    elif r == "C":
                        if book.count[w] - params:
                            book.count[w] &= params
                            changed = True
                    elif r == "N":
                        if book.conn[w] - params:
                            book.conn[w] &= params
                            changed = True
        if verbose:
            g = sum(book.grounded(w) for w in book.roles)
            print(f"  pass {p}: {g}/{len(book.roles)} grounded, "
                  f"{used} conclusive demos used", flush=True)
        if not changed:
            break
    return book


def predict(book, words):
    """Parse + execute with the learned book (must be unambiguous)."""
    results = set()
    for parse in command_parses(book, words):
        try:
            if parse[0] == "one":
                gs = clause_groundings(book, words, parse[1])
                if gs == "UNK":
                    continue
                for o, b in gs:
                    if o is not None:
                        results.add(o)
            else:
                _, i, t1, t2 = parse
                g1 = clause_groundings(book, words[:i], t1)
                g2 = clause_groundings(book, words[i + 1:], t2)
                if g1 == "UNK" or g2 == "UNK":
                    continue
                for swap in book.conn[words[i]]:
                    for o1, _ in g1:
                        for o2, _ in g2:
                            if o1 is not None and o2 is not None:
                                results.add(exec_command([o1, o2], swap))
        except Exception:
            continue
    return results


def evaluate(book, demos, name):
    exact = amb = fail = 0
    for words, out in demos:
        preds = predict(book, words)
        if len(preds) == 1 and next(iter(preds)) == out:
            exact += 1
        elif out in preds:
            amb += 1
        else:
            fail += 1
    n = len(demos)
    print(f"  {name}: exact {100 * exact / n:.1f}  "
          f"(ambiguous-but-contains {amb}, wrong {fail}, n={n})",
          flush=True)
    return exact / n


# --------------------------------------------- oracle schema validation
ORACLE = {  # TEST-ONLY scaffold for pre-registration check #1
    "walk": ("V", ("I_WALK",)), "look": ("V", ("I_LOOK",)),
    "run": ("V", ("I_RUN",)), "jump": ("V", ("I_JUMP",)),
    "turn": ("V", ()),
    "left": ("D", "I_TURN_LEFT"), "right": ("D", "I_TURN_RIGHT"),
    "opposite": ("S", ("pre", 2)), "around": ("S", ("inter", 4)),
    "twice": ("C", 2), "thrice": ("C", 3),
    "and": ("N", 0), "after": ("N", 1)}


def oracle_check(demos):
    ok = 0
    bad = None
    for words, out in demos:
        # split on connector
        conn = [i for i, w in enumerate(words) if ORACLE[w][0] == "N"]
        segs = ([words] if not conn else
                [words[:conn[0]], words[conn[0] + 1:]])
        swap = ORACLE[words[conn[0]]][1] if conn else 0
        outs = []
        for seg in segs:
            verb_em, dir_atom, struct, count = None, None, ("pre", 1), 1
            for w in seg:
                r, p = ORACLE[w]
                if r == "V":
                    verb_em = p
                elif r == "D":
                    dir_atom = p
                elif r == "S":
                    struct = p
                elif r == "C":
                    count = p
            outs.append(exec_clause(verb_em, dir_atom, struct, count))
        pred = exec_command(outs, swap)
        if pred == out:
            ok += 1
        elif bad is None:
            bad = (words, out, pred)
    print(f"  oracle schema check: {ok}/{len(demos)} "
          f"({100 * ok / len(demos):.2f}%)", flush=True)
    if bad:
        print(f"  FIRST MISMATCH: {bad}", flush=True)
    return ok == len(demos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/scan")
    ap.add_argument("--splits", nargs="+",
                    default=["simple", "addprim_jump", "length"])
    args = ap.parse_args()
    t0 = time.time()
    for split in args.splits:
        train = load(f"{args.data}/tasks_train_{split}.txt")
        test = load(f"{args.data}/tasks_test_{split}.txt")
        print(f"\n=== {split}: {len(train)} train / {len(test)} test",
              flush=True)
        if split == "simple":
            oracle_check(train)
        book = learn(Book(), train)
        if book.contradictions:
            print(f"  CONTRADICTIONS: {len(book.contradictions)}; "
                  f"first: {book.contradictions[0]}", flush=True)
        for w in sorted(book.roles):
            r = (next(iter(book.roles[w]))
                 if len(book.roles[w]) == 1 else book.roles[w])
            print(f"    {w:>9s} -> {r} {book.param_candidates(w, r) if isinstance(r, str) else ''}",
                  flush=True)
        evaluate(book, train, "train")
        evaluate(book, test, "TEST")
    print(f"\ntotal {time.time() - t0:.0f}s, zero gradients", flush=True)


if __name__ == "__main__":
    main()
