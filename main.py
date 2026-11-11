#!/usr/bin/env python3
"""werewolf-agent-01 entry: a deterministic, pure-logic Werewolf player.

Python 3.13 standard library only; no model calls, no keys, no clock and no
unseeded randomness in anything sent. Every decision is a pure function of the
one game's view the env serves with it (the view carries the full public log),
so the live run and the replay send byte-identical requests.

Protocol (brief, werewolf-agent-01):
  POST $MAKEFUN_ENV_URL/observe {}      -> {"games", "finished", "done", "pending": [...]}
  POST $MAKEFUN_ENV_URL/act {"game", "decision", "target"|"text"|"pass"} -> 200 once taken
Decisions: "speak" (statement of 1..500 chars after folding, or pass), "vote",
"kill", "protect", "inspect" (target one of the decision's options, or null).

Strategy, in short:
  - Rebuild the public record from the view's log on every decision: role
    claims, claimed seer results, accusations, votes, exiles, night deaths.
  - Cross-check claims against revealed roles: a claimant whose claimed
    result is contradicted by a revealed role is a proven liar; a lone
    uncontradicted claimant is provisionally trusted.
  - Village seats vote a suspicion score built from those signals and say
    why, briefly and consistently.
  - The seer inspects high-information seats and reveals with a full check
    history when it lands a wolf, when a fake seer appears, or late.
  - The doctor guards the trusted seer, else the most trusted voice.
  - A werewolf mirrors the partner's kill pick when it is already in (the
    env shows it), else kills the most dangerous village voice; by day it
    argues like a villager from the same public record, never exposes its
    partner, and buses the partner only when the partner is already doomed.
"""

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
MAX_STATEMENT = 480  # the env's limit is 500 chars after folding; stay under it
WALL_BUDGET_S = 1500  # the trial's is 1,800; leaving only never changes a request

# --------------------------------------------------------------------------
# small helpers


def _h(*parts):
    """A deterministic 64-bit hash of the parts: the entry's only 'randomness'."""
    data = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def det_pick(cands, *key):
    """The candidate with the lowest state-derived hash: stable across runs."""
    return min(cands, key=lambda c: (_h(*(key + (c,))), c))


def post(path, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ENV_URL + path, data=data, method="POST",
        headers={"content-type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            body = json.loads(err.read().decode("utf-8"))
        except Exception:
            body = {}
        return err.code, body


def fold(text):
    """The env's folding: line breaks/tabs to one space, controls dropped, trimmed."""
    text = re.sub(r"[\t\n\v\f\r  ]+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()


def clip(text):
    text = fold(text)
    if len(text) <= MAX_STATEMENT:
        return text
    cut = text[:MAX_STATEMENT]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 200 else cut).strip()


# --------------------------------------------------------------------------
# reading statements: claims, claimed results, accusations, defences

SEER_CLAIM = [re.compile(p) for p in (
    r"\bi(?:'m| am) the seer\b", r"\bi(?:'m| am) seer\b", r"\bi claim seer\b",
    r"\bi claim to be the seer\b", r"\bseer here\b", r"\bas the seer,",
    r"\bi am claiming seer\b", r"\bmy role is seer\b", r"\bi'?m the real seer\b",
)]
DOC_CLAIM = [re.compile(p) for p in (
    r"\bi(?:'m| am) the doctor\b", r"\bi(?:'m| am) doctor\b", r"\bi claim doctor\b",
    r"\bi claim to be the doctor\b", r"\bdoctor here\b", r"\bmy role is doctor\b",
)]
CLEAR_WORDS = re.compile(
    r"not a (werewolf|wolf)|not werewolf|is clean|\bclean\b|innocent|a villager|"
    r"is village|is town|not suspicious|trust")
WOLF_WORDS = re.compile(r"werewolf|is a wolf|is the wolf|\bwolf\b")
ACCUSE = [re.compile(p) for p in (
    r"(?:suspect|accuse|voting for|vote for|vote|exile|lynch)\s+(?:for\s+)?seat\s*(\d+)",
    r"seat\s*(\d+)\s+is\s+(?:suspicious|a wolf|the wolf|scum)",
    r"seat\s*(\d+)\s+is\s+my\s+(?:top\s+)?suspect",
)]


class Belief:
    """Everything the public record supports, rebuilt from one game's view."""

    def __init__(self, view):
        self.view = view
        self.me = view["you"]["seat"]
        self.my_role = view["you"]["role"]
        self.day = view["day"]
        self.players = {p["seat"]: p for p in view["players"]}
        self.names = {s: p["name"] for s, p in self.players.items()}
        self.alive = sorted(s for s, p in self.players.items() if p["alive"])
        self.revealed = {s: p["role"] for s, p in self.players.items() if p["role"]}
        self.speeches = []       # log entries of kind speech/pass, in order
        self.speech_count = {s: 0 for s in self.players}
        self.pass_count = {s: 0 for s in self.players}
        self.votes = []          # (day, voter, target)
        self.exiles = []         # (day, seat, role)
        self.dawns = []          # (day, died seat or None, role or None)
        self.seer_claims = {}    # seat -> first day claimed
        self.doc_claims = {}
        self.claimed = {}        # claimant -> [(day, target, is_wolf)]
        self.accusations = []    # (speaker, target, day)
        self.defences = []       # (speaker, target, day): "seat N is clean"-style
        self.my_speeches = []    # my own speech texts, in order
        self._parse()

    # -- parsing -----------------------------------------------------------

    def _parse(self):
        for e in self.view["log"]:
            kind = e["kind"]
            if kind == "speech":
                self.speeches.append(e)
                self.speech_count[e["seat"]] += 1
                if e["seat"] == self.me:
                    self.my_speeches.append(e["text"])
                self._read_speech(e)
            elif kind == "pass":
                self.speeches.append(e)
                self.pass_count[e["seat"]] += 1
            elif kind == "vote":
                self.votes.append((e["day"], e["seat"], e["target"]))
            elif kind == "exile":
                if e["seat"] is not None:
                    self.exiles.append((e["day"], e["seat"], e["role"]))
            elif kind == "dawn":
                self.dawns.append((e["day"], e["died"], e["role"]))

    def _read_speech(self, e):
        seat, day, text = e["seat"], e["day"], e["text"]
        low = " " + text.lower() + " "
        claims_seer = any(r.search(low) for r in SEER_CLAIM)
        if claims_seer:
            self.seer_claims.setdefault(seat, day)
        if any(r.search(low) for r in DOC_CLAIM):
            self.doc_claims.setdefault(seat, day)
        is_claimant = claims_seer or seat in self.seer_claims
        for m in re.finditer(r"seat\s*(\d+)", low):
            t = int(m.group(1))
            if t == seat or t not in self.players:
                continue
            win = low[max(0, m.start() - 45): m.end() + 65]
            is_clear = bool(CLEAR_WORDS.search(win))
            is_wolf = (not is_clear) and bool(WOLF_WORDS.search(win))
            if is_wolf or is_clear:
                if is_claimant:
                    self.claimed.setdefault(seat, []).append((day, t, is_wolf))
                elif is_wolf:
                    self.accusations.append((seat, t, day))
                else:
                    self.defences.append((seat, t, day))
        if not is_claimant:
            for r in ACCUSE:
                for m in r.finditer(low):
                    t = int(m.group(1))
                    if t != seat and t in self.players:
                        self.accusations.append((seat, t, day))

    # -- claim bookkeeping ---------------------------------------------------

    def claimant_status(self):
        """seat -> 'proven' | 'liar' | 'open' for every seer claimant."""
        out = {}
        revealed_seer = [s for s, r in self.revealed.items() if r == "seer"]
        for c in self.seer_claims:
            status = "open"
            for (_d, t, is_wolf) in self.claimed.get(c, []):
                if t in self.revealed and (self.revealed[t] == "werewolf") != is_wolf:
                    status = "liar"
                if is_wolf and self.revealed.get(t) == "werewolf":
                    status = "proven" if status == "open" else status
            if revealed_seer and c not in revealed_seer:
                status = "liar"
            if self.my_role == "seer" and c != self.me:
                status = "liar"
            out[c] = status
        return out

    def trusted_seer(self):
        """(seat, weight): the lone credible claimant, else (None, 0)."""
        status = self.claimant_status()
        live = {c: s for c, s in status.items()
                if s != "liar" and self.players[c]["alive"]}
        if len(live) == 1:
            c, s = next(iter(live.items()))
            return c, 1.0 if s == "proven" else 0.8
        proven = [c for c, s in live.items() if s == "proven"]
        if len(proven) == 1:
            return proven[0], 1.0
        return None, 0.0

    # -- suspicion -----------------------------------------------------------

    def suspicion(self):
        """(scores, reasons): village-side suspicion per living seat but mine."""
        status = self.claimant_status()
        tseer, tweight = self.trusted_seer()
        scores = {s: 0.0 for s in self.alive if s != self.me}
        reasons = {s: [] for s in scores}

        def add(s, pts, why):
            if s in scores:
                scores[s] += pts
                if why and len(reasons[s]) < 3:
                    reasons[s].append(why)

        # my own inspections are ground truth when I am the seer
        if self.my_role == "seer" and self.view.get("seerResults"):
            for r in self.view["seerResults"]:
                if r["seat"] in scores:
                    if r["werewolf"]:
                        add(r["seat"], 1000, "")
                    else:
                        scores[r["seat"]] = -1000

        # claimed results, weighted by the claimant's credibility
        for c, results in self.claimed.items():
            st = status.get(c, "open")
            if st == "liar":
                for (_d, t, is_wolf) in results:
                    if not is_wolf:
                        add(t, 25, f"cleared by seat {c}, a proven liar")
                continue
            if c == tseer:
                wolf_pts, clear_pts = (85, -70) if tweight == 1.0 else (70, -55)
            else:
                wolf_pts, clear_pts = 30, -20
            for (_d, t, is_wolf) in results:
                if is_wolf:
                    add(t, wolf_pts,
                        f"named a werewolf by seer claimant seat {c}")
                else:
                    add(t, clear_pts, "")

        # proven liars and impossible claims
        for c, st in status.items():
            if st == "liar":
                add(c, 95, f"seat {c}'s seer claim is disproven")
        first_claim_day = min(self.seer_claims.values()) if self.seer_claims else None
        for c, d in self.seer_claims.items():
            if first_claim_day is not None and d > first_claim_day and status.get(c) == "open":
                add(c, 15, "claimed seer only after another claim")
        if self.my_role == "doctor":
            for c in self.doc_claims:
                if c != self.me:
                    add(c, 95, "claims doctor, but I am the doctor")
        revealed_doc = [s for s, r in self.revealed.items() if r == "doctor"]
        for c in self.doc_claims:
            if revealed_doc and c not in revealed_doc:
                add(c, 95, "claims doctor, but the doctor is revealed dead")

        # votes that exiled village-side players, and votes that caught wolves
        exiled_role = {seat: role for (_d, seat, role) in self.exiles}
        for (d, voter, target) in self.votes:
            if target in exiled_role and voter != self.me:
                if exiled_role[target] == "werewolf":
                    add(voter, -12, "")
                else:
                    add(voter, 16, f"voted to exile seat {target}, a {exiled_role[target]}")

        # a night victim's accusations carry their (village) credibility
        victims = {died for (_d, died, role) in self.dawns
                   if died is not None and role != "werewolf"}
        for (sp, t, d) in self.accusations:
            if sp in victims:
                add(t, 12, f"accused by seat {sp}, killed in the night")
            if sp == tseer:
                add(t, 15, "")

        # defences of revealed wolves
        for (sp, t, d) in self.defences:
            if self.revealed.get(t) == "werewolf":
                add(sp, 30, f"defended seat {t}, a werewolf")

        # lurking
        if self.day >= 2:
            for s in scores:
                if self.speech_count.get(s, 0) == 0:
                    add(s, 8, "has said nothing at all")
                elif self.pass_count.get(s, 0) >= 3 and self.speech_count.get(s, 0) <= 1:
                    add(s, 5, "mostly passes")
        return scores, reasons

    # -- today's state ---------------------------------------------------------

    def todays_votes(self):
        return [(voter, t) for (d, voter, t) in self.votes if d == self.day and t is not None]

    def tally(self):
        out = {}
        for (_v, t) in self.todays_votes():
            out[t] = out.get(t, 0) + 1
        return out

    def heat_on_me(self):
        """Distinct living players who accused me in today's debate so far."""
        return {sp for (sp, t, d) in self.accusations
                if t == self.me and d == self.day and self.players[sp]["alive"]}

    def who(self, seat):
        return f"seat {seat} ({self.names[seat]})"


# --------------------------------------------------------------------------
# shared village brain (also the werewolf's camouflage, minus its knowledge)


def village_target(b, scores):
    """The seat the village side should exile: suspicion, with coordination."""
    cands = [s for s in scores]
    if not cands:
        return None
    known = [s for s in cands if scores[s] >= 1000]
    if known:
        return known[0]
    top = max(cands, key=lambda s: (scores[s], -s))
    tally = b.tally()
    # follow a credible bandwagon rather than split the vote into no exile
    for s in cands:
        if tally.get(s, 0) >= 2 and scores[s] >= 25 and scores[s] >= 0.6 * scores[top]:
            if s != top:
                return s
    if scores[top] <= 5 and b.day == 1:
        # nothing to read yet: press a quiet seat, deterministically
        return min(cands, key=lambda s: (b.speech_count.get(s, 0), _h("d1", b.view["you"]["seat"], s), s))
    return top


def top_reasons(reasons, seat, n=2):
    out = []
    for r in reasons.get(seat, []):
        if r and r not in out:
            out.append(r)
    return out[:n]


# --------------------------------------------------------------------------
# the seer


class SeerPlay:
    def __init__(self, b):
        self.b = b
        self.results = b.view.get("seerResults") or []
        self.checked = {r["seat"] for r in self.results}
        self.wolves_found = [r["seat"] for r in self.results if r["werewolf"]]
        self.clears = [r["seat"] for r in self.results if not r["werewolf"]]
        self.revealed = any("i am the seer" in t.lower() or "i'm the seer" in t.lower()
                             for t in b.my_speeches)

    def inspect(self, options):
        b = self.b
        cands = [s for s in options if s not in self.checked
                 and b.revealed.get(s) is None]
        if not cands:
            cands = list(options)
        status = b.claimant_status()
        claimants = [s for s in cands if s in b.seer_claims and status.get(s) != "liar"]
        if claimants:
            return det_pick(claimants, "insp-claim", b.day)
        scores, _ = b.suspicion()
        ranked = sorted(cands, key=lambda s: (-scores.get(s, 0), s))
        if ranked and scores.get(ranked[0], 0) > 20:
            return ranked[0]
        return det_pick(cands, "insp", b.me, b.day)

    def history_text(self):
        parts = []
        for r in sorted(self.results, key=lambda r: r["night"]):
            verdict = "a werewolf" if r["werewolf"] else "not a werewolf"
            parts.append(f"night {r['night']} I checked seat {r['seat']}: {verdict}")
        return "; ".join(parts)

    def should_reveal(self):
        b = self.b
        if self.revealed:
            return False
        live_wolf = any(b.players[s]["alive"] for s in self.wolves_found)
        if live_wolf:
            return True
        if any(c != b.me for c in b.seer_claims):
            return True  # a fake seer is talking: counter with the real history
        if len(b.heat_on_me()) >= 2:
            return True
        if b.day >= 3 and len(self.clears) >= 2:
            return True
        if b.day >= 4 and self.results:
            return True
        return False


# --------------------------------------------------------------------------
# decisions, per role


def decide_speak_village(b, scores, reasons):
    """A short, concrete, consistent statement for a village-side seat."""
    me = b.me
    target = village_target(b, scores)
    tseer, _w = b.trusted_seer()
    status = b.claimant_status()

    if b.my_role == "seer":
        sp = SeerPlay(b)
        if sp.should_reveal():
            wolf = next((s for s in sp.wolves_found if b.players[s]["alive"]), None)
            line = f"I am the seer. My checks: {sp.history_text()}."
            if wolf is not None:
                line += f" Seat {wolf} is a werewolf. Vote seat {wolf} today."
            return clip(line)
        if sp.revealed:
            latest = max(sp.results, key=lambda r: r["night"]) if sp.results else None
            if latest:
                verdict = "a werewolf" if latest["werewolf"] else "not a werewolf"
                line = f"Seer update: last night I checked seat {latest['seat']}: {verdict}."
                if latest["werewolf"] and b.players[latest["seat"]]["alive"]:
                    line += f" Vote seat {latest['seat']}."
                return clip(line)
            return None

    # a fake claimant I can name (I am not the seer, but a claimant is disproven)
    liars = [c for c, s in status.items() if s == "liar" and b.players[c]["alive"] and c != me]
    if liars and target == liars[0]:
        return clip(f"Seat {liars[0]}'s seer claim does not hold up against the revealed "
                    f"roles. I do not trust it. My vote today: seat {liars[0]}.")

    if tseer is not None and tseer != me:
        named = [t for (_d, t, w) in b.claimed.get(tseer, []) if w and b.players[t]["alive"]]
        if named:
            return clip(f"I believe seat {tseer}'s seer claim; nothing contradicts it. "
                        f"Their werewolf is seat {named[0]}. I am voting seat {named[0]} today.")
        cleared = [t for (_d, t, w) in b.claimed.get(tseer, []) if not w]
        if cleared and target is not None and target not in cleared:
            return clip(f"I believe seat {tseer}'s seer claim. Seat {target} is where my "
                        f"vote goes today; the claim and the votes point there.")

    if target is not None:
        rs = top_reasons(reasons, target)
        if rs:
            return clip(f"My main suspect is seat {target}: " + "; ".join(rs) +
                        f". I will vote seat {target} today.")
        if b.day == 1:
            died = b.dawns[-1][1] if b.dawns else None
            lost = f"We lost seat {died} last night. " if died is not None else ""
            return clip(f"No hard information yet. {lost}I am watching "
                        f"seat {target} and will follow the strongest case today.")
        return clip(f"I have no claim to make. On the record so far, seat {target} is "
                    f"where my vote goes today.")
    return None


def speak_round2_needed(b, prev_text):
    """Round 2 is for answering new claims, heat, or a changed picture."""
    if b.my_role == "seer" and SeerPlay(b).should_reveal():
        return True
    if len(b.heat_on_me()) >= 1 and prev_text is not None:
        return True
    return False


def decide_speak(b):
    scores, reasons = b.suspicion()
    if b.my_role == "werewolf":
        return decide_speak_wolf(b, scores, reasons)
    # which round am I speaking in? count my speeches today
    said_today = [e for e in b.speeches
                  if e["day"] == b.day and e["seat"] == b.me and e["kind"] == "speech"]
    if said_today:
        # round 2: only speak again when there is something to answer
        if not speak_round2_needed(b, said_today[-1]["text"]):
            return None
        if len(b.heat_on_me()) >= 1 and b.my_role in ("seer",):
            sp = SeerPlay(b)
            if sp.should_reveal():
                wolf = next((s for s in sp.wolves_found if b.players[s]["alive"]), None)
                line = f"I am the seer, whatever seat claims say. My checks: {sp.history_text()}."
                if wolf is not None:
                    line += f" Vote seat {wolf}."
                return clip(line)
        if len(b.heat_on_me()) >= 1:
            target = village_target(b, scores)
            return clip(f"I am on the village side; my record is consistent. The case "
                        f"against me is thinner than the case against seat {target}. "
                        f"My vote stays seat {target}.")
        return None
    return decide_speak_village(b, scores, reasons)


def decide_vote(b, options):
    scores, _reasons = b.suspicion()
    if b.my_role == "werewolf":
        return decide_vote_wolf(b, scores, options)
    if b.my_role == "seer":
        sp = SeerPlay(b)
        live = [s for s in sp.wolves_found if s in options]
        if live:
            return live[0]
    target = village_target(b, scores)
    if target in options:
        return target
    cands = [s for s in options if scores.get(s, 0) > -100]
    if cands:
        return max(cands, key=lambda s: (scores.get(s, 0), -s))
    return det_pick(list(options), "vote", b.me, b.day) if options else None


# --------------------------------------------------------------------------
# the doctor


def decide_protect(b, options):
    tseer, _w = b.trusted_seer()
    if tseer is not None and tseer in options:
        return tseer
    scores, _ = b.suspicion()
    trusted = [s for s in options if s != b.me]
    if not trusted:
        return None
    # guard the least suspected living voice (never the likeliest wolf)
    return min(trusted, key=lambda s: (scores.get(s, 0), _h("prot", b.me, b.day, s), s))


# --------------------------------------------------------------------------
# the werewolf


def wolf_priorities_kill(b, options):
    """Kill order: the seer who named us, a credible seer, then our accusers,
    then the village's most trusted voice, then its loudest."""
    partner = b.view["you"].get("partner") or {}
    pseat = partner.get("seat")
    status = b.claimant_status()
    opts = [s for s in options if s != pseat]
    if not opts:
        return None
    # a claimant who named me or my partner is the real seer until disproven
    for c, results in b.claimed.items():
        if c in opts and status.get(c) != "liar":
            if any(w and t in (b.me, pseat) for (_d, t, w) in results):
                return c
    tseer, _w = b.trusted_seer()
    if tseer in opts:
        return tseer
    claimants = [s for s in opts if s in b.seer_claims and status.get(s) != "liar"]
    if claimants:
        return det_pick(claimants, "kill-claim", b.day)
    doc = [s for s in opts if s in b.doc_claims]
    if doc:
        return det_pick(doc, "kill-doc", b.day)
    accusers = {}
    for (sp, t, _d) in b.accusations:
        if t in (b.me, pseat) and sp in opts:
            accusers[sp] = accusers.get(sp, 0) + 1
    if accusers:
        return max(accusers, key=lambda s: (accusers[s], -s))
    scores, _ = b.suspicion()
    trusted = min(opts, key=lambda s: (scores.get(s, 0), s))
    if scores.get(trusted, 0) < 0:
        return trusted
    loudest = max(opts, key=lambda s: (b.speech_count.get(s, 0), -s))
    return loudest


def decide_kill(b, options):
    # if my partner already named someone tonight, agree: the kill needs both
    partner = b.view["you"].get("partner") or {}
    for v in (b.view.get("wolfVotes") or []):
        if v["night"] == b.day and v["seat"] == partner.get("seat") and v["target"] in options:
            return v["target"]
    return wolf_priorities_kill(b, options)


def partner_doomed(b, scores):
    partner = b.view["you"].get("partner") or {}
    pseat = partner.get("seat")
    if pseat is None or not b.players[pseat]["alive"]:
        return False
    tseer, _w = b.trusted_seer()
    if tseer is not None and any(w and t == pseat for (_d, t, w) in b.claimed.get(tseer, [])):
        return True
    tally = b.tally()
    return tally.get(pseat, 0) >= 3


def decide_vote_wolf(b, scores, options):
    partner = b.view["you"].get("partner") or {}
    pseat = partner.get("seat")
    tally = b.tally()
    if pseat in options and partner_doomed(b, scores):
        return pseat  # bus: the partner is going down; be seen on the right side
    cands = [s for s in options if s != pseat]
    if not cands:
        return None
    # push the exile the village is already building against a non-wolf
    return max(cands, key=lambda s: (scores.get(s, 0) + 12 * tally.get(s, 0), -s))


def decide_speak_wolf(b, scores, reasons):
    me = b.me
    partner = b.view["you"].get("partner") or {}
    pseat = partner.get("seat")
    said_today = [e for e in b.speeches
                  if e["day"] == b.day and e["seat"] == me and e["kind"] == "speech"]
    if said_today:
        return None  # one statement a day is villager-like; do not over-talk
    # named by a seer claimant: deny hard and counter-claim with a clean,
    # fabricated history anchored on revealed (dead, non-wolf) seats.
    status = b.claimant_status()
    for c, results in b.claimed.items():
        if status.get(c) == "liar":
            continue
        if any(w and t == me for (_d, t, w) in results):
            dead_village = [s for s, r in b.revealed.items() if r != "werewolf"]
            nights = list(range(1, b.day + 1))  # night n precedes day n
            hist = []
            for i, s in enumerate(dead_village[: max(0, len(nights) - 1)]):
                hist.append(f"night {nights[i]} I checked seat {s}: not a werewolf")
            hist.append(f"night {nights[-1]} I checked seat {c}: a werewolf")
            return clip(f"Seat {c} is lying. I am the seer. My checks: " +
                        "; ".join(hist) + f". Seat {c} named me because I found them. "
                        f"Vote seat {c}.")
    if pseat is not None and partner_doomed(b, scores):
        return clip(f"The case against seat {pseat} is hard to argue with now. "
                    f"I will vote seat {pseat} today.")
    # ordinary cover: argue the village case against a non-wolf, from the record
    pub_scores = dict(scores)
    if pseat in pub_scores:
        pub_scores[pseat] = -999  # never build the case against my partner
    target = village_target(b, {s: v for s, v in pub_scores.items() if s != pseat})
    if target is None or target == pseat:
        return None
    rs = top_reasons(reasons, target)
    if rs:
        return clip(f"I read the record the same way: seat {target} stands out — " +
                    "; ".join(rs) + f". My vote: seat {target}.")
    if b.day == 1:
        return clip(f"Rough night. I have no claim; I am watching how the votes "
                    f"line up and I lean seat {target} for now.")
    return clip(f"No claim from me. On the whole record, seat {target} is my vote today.")


# --------------------------------------------------------------------------
# the runner


def decide(pending):
    view = pending["view"]
    kind = pending["decision"]
    options = pending.get("options") or []
    b = Belief(view)
    if kind == "speak":
        text = decide_speak(b)
        if text:
            return {"game": pending["game"], "decision": "speak", "text": text}
        return {"game": pending["game"], "decision": "speak", "pass": True}
    if kind == "vote":
        target = decide_vote(b, options)
    elif kind == "kill":
        target = decide_kill(b, options)
    elif kind == "protect":
        target = decide_protect(b, options)
    elif kind == "inspect":
        target = SeerPlay(b).inspect(options) if options else None
    else:
        target = None
    if target not in options:
        target = None
    return {"game": pending["game"], "decision": kind, "target": target}


def fallback(pending):
    if pending["decision"] == "speak":
        return {"game": pending["game"], "decision": "speak", "pass": True}
    return {"game": pending["game"], "decision": pending["decision"], "target": None}


def main():
    if not ENV_URL:
        print("MAKEFUN_ENV_URL is not set", file=sys.stderr)
        return 1
    started = time.monotonic()
    seen = set()
    while time.monotonic() - started < WALL_BUDGET_S:
        status, obs = post("/observe", {})
        if status != 200 or not isinstance(obs, dict):
            time.sleep(1)
            continue
        if obs.get("done"):
            return 0
        for pending in obs.get("pending") or []:
            key = (pending["game"], pending["decision"],
                   len(pending["view"].get("log") or []))
            if key in seen:
                continue
            seen.add(key)
            try:
                act = decide(pending)
            except Exception:
                act = fallback(pending)
            status, _ = post("/act", act)
            if status in (400, 409) and act != fallback(pending):
                post("/act", fallback(pending))
    return 0


if __name__ == "__main__":
    sys.exit(main())
