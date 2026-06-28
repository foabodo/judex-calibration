#!/usr/bin/env python3
"""Phase 0 — FREE accuracy pre-check (no API calls, no GPU).

Gate for Study A. The existing AIReg-Bench LLM annotations (the *paper's* raters,
a PROXY — NOT our six pre/post annotators, which have never been run on AIReg)
are scored for argmax accuracy against the human-reconciled GT mode. If even
frontier models cannot pick the modal human compliance level often enough, the
accuracy gate is structural to the task and any paid base-model serving is
premature.

stdlib only.
"""
import json, re, zipfile, glob, os
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = "/Users/fabodo/Downloads/Projects/judex/judex-evaluator"
AIREG = f"{ROOT}/data/external/aireg_bench"
GT_REPORT = f"{ROOT}/runs/stage9-gemini-gpt-medium/metrics_report.json"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def col_letter(idx0):  # 0->A, 1->B, ... 26->AA
    s = ""
    n = idx0
    while True:
        s = chr(ord("A") + n % 26) + s
        n = n // 26 - 1
        if n < 0:
            break
    return s


def read_grid(path):
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si"):
            ss.append("".join(t.text or "" for t in si.iter(NS + "t")))
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    grid = {}
    for c in sheet.iter(NS + "c"):
        ref, t, v = c.get("r"), c.get("t"), c.find(NS + "v")
        if v is None:
            continue
        grid[ref] = ss[int(v.text)] if t == "s" else v.text
    return grid


def gt_mode_by_label():
    m = json.load(open(GT_REPORT))
    return {it["item_label"]: it["ground_truth_argmax_index"] + 1 for it in m["items"]}  # 1..5


REF = f"{AIREG}/llm_annotations/4o_annotations.xlsx"  # a LABELLED workbook


def canonical_labels():
    """Row order (Art N / Scenario X) from a labelled workbook; data starts row 3."""
    g = read_grid(REF)
    labels, row = [], 3
    while g.get(f"A{row}"):
        labels.append(g[f"A{row}"])
        row += 1
    return labels  # 15 = 5 articles x 3 scenarios


def model_ratings(path, canonical):
    """Return {item_label: rating 1..5}. Some workbooks omit row/col labels, so map
    POSITIONALLY: row 3+i -> canonical[i]; Compliance[1-5] cols are the 1st of each
    4-col Use group (B,F,J,...). Validated against the labelled workbooks below."""
    g = read_grid(path)
    use_cols = {u: col_letter(1 + u * 4) for u in range(8)}
    out = {}
    for i, label in enumerate(canonical):
        row = 3 + i
        for u, cc in use_cols.items():
            val = g.get(f"{cc}{row}")
            if val is None:
                continue
            try:
                r = int(round(float(val)))
            except ValueError:
                continue
            if 1 <= r <= 5:
                out[f"{label} | Use {u + 1}"] = r
    return out


def main():
    gt = gt_mode_by_label()
    canonical = canonical_labels()
    files = sorted(glob.glob(f"{AIREG}/llm_annotations/*_annotations.xlsx"))
    rows = []
    for f in files:
        name = os.path.basename(f).replace("_annotations.xlsx", "")
        pred = model_ratings(f, canonical)
        joined = [(pred[k], gt[k]) for k in pred if k in gt]
        n = len(joined)
        if not n:
            rows.append((name, 0, 0.0, 0.0, 0.0)); continue
        exact = sum(p == t for p, t in joined) / n
        within1 = sum(abs(p - t) <= 1 for p, t in joined) / n
        mae = sum(abs(p - t) for p, t in joined) / n
        rows.append((name, n, exact, within1, mae))
    rows.sort(key=lambda r: -r[2])
    print(f"GT cells: {len(gt)}  |  models: {len(files)}  |  PROXY (AIReg-Bench raters, not our six)\n")
    print(f"{'model':14s} {'n':>4s} {'argmax_acc':>11s} {'within±1':>9s} {'MAE':>6s}")
    print("-" * 50)
    for name, n, ex, w1, mae in rows:
        print(f"{name:14s} {n:4d} {ex:11.3f} {w1:9.3f} {mae:6.3f}")
    accs = [r[2] for r in rows if r[1]]
    print("-" * 50)
    print(f"best={max(accs):.3f}  median={sorted(accs)[len(accs)//2]:.3f}  worst={min(accs):.3f}")
    out = {"gt_cells": len(gt), "proxy_note": "AIReg-Bench paper raters, NOT the six pre/post annotators",
           "models": [{"model": n, "n": nn, "argmax_acc": ex, "within1_acc": w1, "mae": mae}
                      for n, nn, ex, w1, mae in rows]}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "runs", "phase0"), exist_ok=True)
    dst = os.path.join(os.path.dirname(__file__), "..", "runs", "phase0", "accuracy_precheck.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(dst)}")


if __name__ == "__main__":
    main()
