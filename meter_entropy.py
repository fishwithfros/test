#!/usr/bin/env python3
"""
meter_entropy.py — standalone cross-entropy model of rhythm.

Aligns a passage's beat-pattern (via DB word lookups) against a proposed
meter tiled to matching length, then scores per-beat surprisal of the
passage under a smoothed periodic model derived from the meter. The
substitution/gap cost tables act as the surprisal kernel:

    P(event at position i) ∝ exp(-cost(event, meter[i mod k]) / T)

so cost ≈ T·ln2·surprisal(bits). Output: total cross-entropy (bits/beat),
a per-beat surprisal trace, and optional ranking of candidate meters.

Usage:
  python3 meter_entropy.py --meter "da DUM da DUM" --words "the road goes ever on"
  python3 meter_entropy.py --meters meters.txt --words "..."   # rank candidates
"""

import argparse
import math
import sys
import duckdb

# ---------------------------------------------------------------- symbols

ATOMS = ["da", "DA", "dum", "DUM", "daaa", "daa", "DUMMM", "DUMM",
         "da-", "DUM-", "DA-", "dum-"]

PROPS = {
    "da":    ("weak",   "short"),
    "DA":    ("accent", "short"),
    "dum":   ("half",   "short"),
    "DUM":   ("strong", "short"),
    "daaa":  ("weak",   "long"),
    "daa":   ("weak",   "long"),
    "DUMMM": ("strong", "long"),
    "DUMM":  ("strong", "long"),
    "da-":   ("weak",   "clipped"),
    "DUM-":  ("strong", "clipped"),
    "DA-":   ("accent", "clipped"),
    "dum-":  ("half",   "clipped"),
}

STRESS_RANK = {"weak": 0, "accent": 1, "half": 2, "strong": 3}
UNKNOWN = ("weak", "short")


def tokenize(pattern):
    """Split a pattern string into atomic beat tokens (hyphen = clipped)."""
    out = []
    for chunk in pattern.replace("|", " ").split():
        parts = chunk.split("-")
        for k, t in enumerate(parts):
            t = t.strip()
            if not t:
                continue
            clipped = (k < len(parts) - 1)
            tok = t + "-" if clipped and (t + "-") in PROPS else t
            out.append(tok)
    return out

# ---------------------------------------------------------------- costs

SUB_STRESS_UNIT = 3.0
LICENSED_DEMOTION = 0.5
SUB_DURATION = 1.0
DEL_WEAK, DEL_STRONG = 2.0, 6.0
INS_WEAK, INS_STRONG = 4.0, 10.0
ALT_PATTERN_PENALTY = 0.4
INTERIOR_HOLE_SURCHARGE = 0.05


def props(tok):
    return PROPS.get(tok, UNKNOWN)


def sub_cost(a, b):
    if a == b:
        return 0.0
    sa, da_ = props(a)
    sb, db = props(b)
    ra, rb = STRESS_RANK[sa], STRESS_RANK[sb]
    if sa == "half" and sb == "weak":
        c = LICENSED_DEMOTION
    else:
        c = abs(ra - rb) * SUB_STRESS_UNIT
    if da_ != db:
        pair = {da_, db}
        c += LICENSED_DEMOTION if pair == {"clipped", "short"} else SUB_DURATION
    if c == 0.0:
        c = 0.5
    return c


def gap_cost(tok, kind):
    strong = STRESS_RANK[props(tok)[0]] >= 2
    if kind == "del":
        return DEL_STRONG if strong else DEL_WEAK
    return INS_STRONG if strong else INS_WEAK

# ---------------------------------------------------------------- model
#
# At each target position with meter atom t, the event space is:
#   emit atom a   (12 options), weight exp(-sub_cost(a,t)/T)
#   hole (del)                , weight exp(-gap_cost(t,'del')/T)
# normalized to a distribution; surprisal = -log2 P(event).
# Insertions are boundary events priced against a per-boundary
# "continue" mass: P(ins a) = exp(-gap_cost(a,'ins')/T) / Z_ins where
# Z_ins = 1 + sum over atoms of exp(-gap_cost/T)  (the 1 = no insertion).

LN2 = math.log(2.0)


class PeriodicModel:
    """Smoothed periodic surprisal model for one meter bar."""

    def __init__(self, bar_tokens, temperature=1.0):
        self.bar = bar_tokens
        self.T = temperature
        self._pos = {}      # target atom -> (dict atom->surprisal, hole surprisal)
        zi = 1.0 + sum(math.exp(-gap_cost(a, "ins") / self.T) for a in ATOMS)
        self._ins = {a: -math.log2(math.exp(-gap_cost(a, "ins") / self.T) / zi)
                     for a in ATOMS}
        self._nogap = -math.log2(1.0 / zi)   # surprisal of "no insertion"

    def _table(self, t):
        if t not in self._pos:
            w = {a: math.exp(-sub_cost(a, t) / self.T) for a in ATOMS}
            wh = math.exp(-gap_cost(t, "del") / self.T)
            z = sum(w.values()) + wh
            self._pos[t] = ({a: -math.log2(v / z) for a, v in w.items()},
                            -math.log2(wh / z))
        return self._pos[t]

    def emit_surprisal(self, atom, target_atom):
        tab, _ = self._table(target_atom)
        # unknown atoms fall back to the weak/short class representative
        return tab.get(atom, tab["da"])

    def hole_surprisal(self, target_atom):
        return self._table(target_atom)[1]

    def ins_surprisal(self, atom):
        return self._ins.get(atom, self._ins["da"])

    @property
    def baseline(self):
        """Surprisal per beat of a perfectly conforming passage."""
        tot = 0.0
        for t in self.bar:
            tot += self.emit_surprisal(t, t) + self._nogap
        return tot / len(self.bar)

# ---------------------------------------------------------------- lattice
# (same alignment method as the map tool: word arcs from DB patterns,
# outer DP over words, inner edit DP per arc)


class Lexicon:
    def __init__(self, db_path):
        self.con = duckdb.connect(db_path, read_only=True)
        tables = [r[0] for r in self.con.execute("SHOW TABLES").fetchall()]
        for cand in ("word_durations", "words"):
            if cand in tables:
                self.table = cand
                break
        else:
            raise SystemExit("No word table found in %s" % db_path)
        self._cache = {}

    def arcs(self, word):
        """[(label, tokens, intrinsic_cost)]"""
        w = word.lower()
        if w in self._cache:
            return self._cache[w]
        rows = self.con.execute(
            "SELECT pattern_1, pattern_2, pattern_3, pattern_4, pattern_5 "
            "FROM %s WHERE word = ?" % self.table, [w]).fetchall()
        arcs = []
        if rows:
            for i, p in enumerate(rows[0]):
                if p and str(p).strip() and str(p) != "nan":
                    toks = tokenize(str(p))
                    if toks:
                        arcs.append((word, toks, i * ALT_PATTERN_PENALTY))
        if not arcs:
            arcs = [(word + "(?)", ["da"], 0.0)]
        self._cache[w] = arcs
        return arcs

    def syllables(self, word):
        row = self.con.execute(
            "SELECT syllables FROM %s WHERE word = ?" % self.table,
            [word.lower()]).fetchone()
        return int(row[0]) if row and row[0] else max(1, len(word) // 3)


def align_arc(tokens, target, start_costs):
    T = len(target)
    n = len(tokens)
    INF = float("inf")
    dp = [[INF] * (T + 1) for _ in range(n + 1)]
    bp = [[None] * (T + 1) for _ in range(n + 1)]
    for j in range(T + 1):
        dp[0][j] = start_costs[j]
    for i in range(n + 1):
        for j in range(T + 1):
            cur = dp[i][j]
            if cur == INF:
                continue
            if i < n and j < T:
                c = cur + sub_cost(tokens[i], target[j])
                if c < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = c
                    bp[i + 1][j + 1] = (i, j, "match", tokens[i], target[j])
            if i < n:
                c = cur + gap_cost(tokens[i], "ins")
                if c < dp[i + 1][j]:
                    dp[i + 1][j] = c
                    bp[i + 1][j] = (i, j, "ins", tokens[i], None)
            if j < T:
                c = cur + gap_cost(target[j], "del") + INTERIOR_HOLE_SURCHARGE
                if c < dp[i][j + 1]:
                    dp[i][j + 1] = c
                    bp[i][j + 1] = (i, j, "del", None, target[j])
    return dp, bp


def align(lex, words, target):
    """Returns (total_cost, path) where path = [(word, label, ops)] and
    ops = [(op, cand_atom, target_atom)] in order."""
    T = len(target)
    INF = float("inf")
    costs = [0.0] + [INF] * T
    for j in range(T):
        c = costs[j] + gap_cost(target[j], "del") + INTERIOR_HOLE_SURCHARGE
        if c < costs[j + 1]:
            costs[j + 1] = c
    trace = [[None] * (T + 1)]

    for w, word in enumerate(words):
        new_costs = [INF] * (T + 1)
        new_trace = [None] * (T + 1)
        for label, toks, intrinsic in lex.arcs(word):
            entry = [c + intrinsic if c < INF else INF for c in costs]
            dp, bp = align_arc(toks, target, entry)
            n = len(toks)
            for j in range(T + 1):
                if dp[n][j] < new_costs[j]:
                    ops = []
                    i, jj = n, j
                    while bp[i][jj] is not None:
                        pi, pj, op, ct, tt = bp[i][jj]
                        ops.append((op, ct, tt))
                        i, jj = pi, pj
                    ops.reverse()
                    new_costs[j] = dp[n][j]
                    new_trace[j] = (jj, label, ops)
        surcharge = INTERIOR_HOLE_SURCHARGE if w < len(words) - 1 else 0.0
        for j in range(T):
            if new_costs[j] < INF:
                c = new_costs[j] + gap_cost(target[j], "del") + surcharge
                if c < new_costs[j + 1]:
                    prev = new_trace[j]
                    new_costs[j + 1] = c
                    new_trace[j + 1] = (prev[0], prev[1],
                                        prev[2] + [("del", None, target[j])])
        costs = new_costs
        trace.append(new_trace)

    if costs[T] == INF:
        return None
    path = []
    j = T
    for w in range(len(words), 0, -1):
        prev_j, label, ops = trace[w][j]
        path.append((words[w - 1], label, ops))
        j = prev_j
    path.reverse()
    if j > 0:
        path.insert(0, ("(lead-in)", "—",
                        [("del", None, target[k]) for k in range(j)]))
    return costs[T], path

# ---------------------------------------------------------------- entropy


def score(model, path):
    """Walk the aligned path, accruing surprisal per event.
    Returns (bits_per_beat, beats, trace) where trace is a list of
    (word, op, cand_atom, target_atom, surprisal_bits)."""
    trace = []
    total = 0.0
    beats = 0
    for word, _label, ops in path:
        for op, ca, ta in ops:
            if op == "match":
                s = model.emit_surprisal(ca, ta) + model._nogap
                beats += 1
            elif op == "del":
                s = model.hole_surprisal(ta) + model._nogap
                beats += 1
            else:  # ins
                s = model.ins_surprisal(ca)
                beats += 1
            total += s
            trace.append((word, op, ca, ta, s))
    return (total / beats if beats else 0.0), beats, trace


def tile(bar, lex, words):
    syl = sum(lex.syllables(w) for w in words)
    bars = max(1, math.ceil(syl / len(bar)))
    return bar * bars, bars


def analyze(lex, words, meter_str, temperature):
    bar = tokenize(meter_str)
    if not bar:
        raise SystemExit("Empty meter pattern: %r" % meter_str)
    target, bars = tile(bar, lex, words)
    res = align(lex, words, target)
    if res is None:
        return None
    cost, path = res
    model = PeriodicModel(bar, temperature)
    bpb, beats, trace = score(model, path)
    return {
        "meter": meter_str, "bar": bar, "bars": bars, "beats": beats,
        "align_cost": cost, "bits_per_beat": bpb,
        "baseline": model.baseline, "excess": bpb - model.baseline,
        "trace": trace, "path": path,
    }

# ---------------------------------------------------------------- output


def print_report(r, show_trace=True, bar_len=None):
    print("meter: %s   (tiled x%d, %d beats)" % (r["meter"], r["bars"], r["beats"]))
    print("cross-entropy: %.3f bits/beat   baseline: %.3f   excess: %.3f"
          % (r["bits_per_beat"], r["baseline"], r["excess"]))
    if not show_trace:
        return
    print()
    print("%-4s %-16s %-6s %-7s %-7s %s" %
          ("beat", "word", "op", "cand", "target", "surprisal(bits)"))
    beat = 0
    for word, op, ca, ta, s in r["trace"]:
        beat += 1
        flag = ""
        if s - (r["baseline"]) > 2.0:
            flag = "  <-- hot"
        print("%-4d %-16s %-6s %-7s %-7s %7.3f%s" %
              (beat, word, op, ca or "—", ta or "—", s, flag))
        if bar_len and beat % bar_len == 0:
            print("     " + "-" * 46)


def round_sig(x):
    """Format surprisal to one decimal place."""
    return "%.1f" % x


DUR_DIGIT = {"clipped": "8", "short": "4", "long": "2"}


def hole_mark(tt):
    s, d = props(tt)
    return "[" + DUR_DIGIT.get(d, "") + ("\u2191" if STRESS_RANK[s] >= 2 else "\u2193") + "]"


def plain_sentence(path, bar_len=None, model=None):
    """Annotated sentence: arrows = stress must be raised (↑) / lowered (↓)
    to the target; digit = required note value on duration mismatch.
    Unfilled target beats (d↕) fold into the following word's bracket;
    trailing unfilled beats at passage end are dropped. With a model, each
    word carries a leading bracket: [p] = passed clean, else
    [holes..., surprisal_bits, needed target atoms]."""
    pieces, beat = [], 0
    pending = []   # hole descriptors awaiting the next word

    def tick(k=1):
        nonlocal beat
        for _ in range(k):
            beat += 1
            if bar_len and beat % bar_len == 0:
                pieces.append("|")

    def hole_desc(tt):
        s, d = props(tt)
        return DUR_DIGIT.get(d, "") + ("\u2191" if STRESS_RANK[s] >= 2
                                       else "\u2193")

    for orig, label, ops in path:
        consumed, over = 0, 0
        pre, post, marks = [], [], []
        tts, bits = [], 0.0
        for op, ca, ta in ops:
            if op == "match":
                consumed += 1
                tts.append(ta)
                if model:
                    bits += model.emit_surprisal(ca, ta) + model._nogap
                if ca != ta:
                    (sa, da_), (sb, db) = props(ca), props(ta)
                    stress_lic = (sa == "half" and sb == "weak")
                    dur_lic = ({da_, db} <= {"clipped", "short"})
                    m = ""
                    if da_ != db and not dur_lic:
                        m += DUR_DIGIT.get(db, "")
                    if sa != sb and not stress_lic:
                        m += "\u2191" if STRESS_RANK[sb] > STRESS_RANK[sa] \
                             else "\u2193"
                    if m:
                        marks.append(m)
            elif op == "ins":
                consumed += 1
                over += 1
                if model:
                    bits += model.ins_surprisal(ca)
            elif op == "del":
                (pre if consumed == 0 else post).append(ta)
        for ta in pre:
            pending.append(hole_desc(ta))
            tick()
        if orig != "(lead-in)":
            word = "".join(marks) + label
            if over:
                word += "(+%d)" % over
            passed = not marks and not over and "(?)" not in label
            if model:
                inner = list(pending)
                if passed and not inner:
                    inner = ["p"]
                elif passed:
                    inner.append("p")
                else:
                    inner += [round_sig(bits), " ".join(tts) or "\u2014"]
                word = "[" + ", ".join(inner) + "]" + word
            elif pending:
                word = "[" + ", ".join(pending) + "]" + word
            pending = []
            pieces.append(word)
            tick(max(0, consumed - over))
        for ta in post:
            pending.append(hole_desc(ta))
            tick()
    # trailing unfilled beats are dropped (pending discarded)
    if pieces and pieces[-1] == "|":
        pieces.pop()
    return " ".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meter", help="proposed meter pattern (one bar)")
    ap.add_argument("--meters", help="file of candidate meters, one per line; "
                                     "ranked by cross-entropy")
    ap.add_argument("--words", required=True)
    ap.add_argument("--db", default="meter.duckdb")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="strictness knob T; cost = T*ln2*surprisal")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--plain", action="store_true",
                    help="emit annotated sentence instead of surprisal table")
    args = ap.parse_args()
    if not args.meter and not args.meters:
        ap.error("provide --meter or --meters")

    lex = Lexicon(args.db)
    words = [w.strip('.,;:!?"\'()[]—–') for w in args.words.split()]
    words = [w for w in words if w]

    if args.meter:
        r = analyze(lex, words, args.meter, args.temperature)
        if r is None:
            sys.exit("No alignment found.")
        if args.plain:
            m = PeriodicModel(r["bar"], args.temperature)
            print(plain_sentence(r["path"], bar_len=len(r["bar"]), model=m))
        else:
            print_report(r, show_trace=not args.no_trace, bar_len=len(r["bar"]))
    else:
        meters = [ln.strip() for ln in open(args.meters) if ln.strip()
                  and not ln.startswith("#")]
        results = []
        for m in meters:
            r = analyze(lex, words, m, args.temperature)
            if r:
                results.append(r)
        results.sort(key=lambda r: r["excess"])
        print("%-32s %10s %10s %10s" %
              ("meter", "bits/beat", "baseline", "excess"))
        for r in results:
            print("%-32s %10.3f %10.3f %10.3f" %
                  (r["meter"], r["bits_per_beat"], r["baseline"], r["excess"]))


if __name__ == "__main__":
    main()
