import json, os, sys
import numpy as np, torch
d = os.path.expanduser("~/.cache/jev_nx/laya")
sys.path.insert(0, d)
from rl_common import QTYPES, build_sequence, collate_items, render_options, temp_bucket
from rl_agent_api import RLAgent

agent = RLAgent(d, device="cpu")

long_state = " ".join(f"Paragraph {i}: the export job for tenant {i} finished with {i * 7 % 13} warnings and no errors." for i in range(120))
many = {f"team_{i:02d}": f"Team {i} handles domain {i} problems, escalations, and paperwork of kind {i}" for i in range(40)}

cases = [
  {"name": "triage",
   "state": {"title": "App crashes on launch", "body": "Since 2.3.1 the app closes immediately on iOS 17."},
   "questions": {
     "kind": {"type": "choice", "instructions": "What kind of issue?", "criteria": {"bug": "Broken", "feature": "New behavior", "other": None}},
     "severity": {"type": "score", "instructions": "How severe?", "criteria": ["Cosmetic", "Workaround", "Blocks", "Data loss"]},
     "security": {"type": "noul", "instructions": "Is this a vulnerability?"}}},
  {"name": "structured",
   "state": {"from": "user@acme.com", "subject": "Duplicate charge", "body": "We were billed twice for March. Please refund the duplicate."},
   "questions": {
     "department": {"type": "choice", "instructions": {"question": "Which team should handle this?", "note": "Pick one"},
                    "criteria": {"billing": {"what": "Charges, refunds", "examples": ["Charged twice"]}, "shipping": {"what": "Delivery status"}, "other": None}},
     "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?", "criteria": {"true": "Money back is requested", "false": "No money is mentioned"}}}},
  {"name": "long_state", "state": long_state,
   "questions": {"errors": {"type": "noul", "instructions": "Did any job report errors?"},
                 "tone": {"type": "score", "instructions": "How alarming is this log?", "criteria": ["calm", "notable", "alarming"]}}},
  {"name": "many_options", "state": "The invoice PDF is blank after the last deploy.",
   "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": dict(sorted(many.items()))}}},
]

def sort_keys(v):
    if isinstance(v, dict): return {k: sort_keys(v[k]) for k in sorted(v)}
    if isinstance(v, list): return [sort_keys(x) for x in v]
    return v

out = []
for case in cases:
    # Jev encodes maps with sorted keys, which is the order a Python server sees.
    case = sort_keys(case)
    qs = {qid: {**q, "criteria": (dict(sorted(q["criteria"].items())) if isinstance(q.get("criteria"), dict) and q["type"] == "choice" else q.get("criteria"))} for qid, q in case["questions"].items()}
    ids_list, items = list(qs.keys()), []
    seqs = {}
    for qid in ids_list:
        q = agent._to_internal(qs[qid])
        seq, markers = build_sequence(agent.tok, case["state"], q, agent.cfg["max_len"], agent.cfg["head_max_len"])
        seqs[qid] = {"ids": seq, "markers": markers, "options": render_options(q)}
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": [0.0] * len(markers), "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "api"})
    b = collate_items([items], agent.tok.pad_token_id)
    with torch.no_grad():
        logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    logits = logits.float().numpy()
    answers = {}
    for r, qid in enumerate(ids_list):
        q = agent._to_internal(qs[qid]); k = len(items[r]["markers"]); qt = QTYPES[q["t"]]
        raw = logits[r, :k]
        z = raw / agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
        p = np.exp(z - z.max()); p = p / p.sum()
        a = {"type": q["t"], "logits": raw.tolist()}
        if q["t"] == "choice":
            keys = list(q["crit"].keys()); a.update(choice=keys[int(p.argmax())], probabilities={kk: float(v) for kk, v in zip(keys, p)})
        elif q["t"] == "score":
            a.update(score=float((np.arange(k) * p).sum()), probabilities={str(i): float(v) for i, v in enumerate(p)})
        else:
            a.update(noul=float(p[1]))
        answers[qid] = a
    out.append({"name": case["name"], "state": case["state"], "questions": qs, "sequences": seqs, "answers": answers, "input_tokens": int(b["attention_mask"].sum())})
    print(case["name"], {k: (v.get("choice") or v.get("score") or v.get("noul")) for k, v in answers.items()}, file=sys.stderr)

json.dump(out, open("laya_golden.json", "w"), ensure_ascii=False, indent=1)
