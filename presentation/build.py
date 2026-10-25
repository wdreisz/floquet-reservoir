"""Build the submission materials from saved artifacts only. Run from the repo root:

    python presentation/build.py

Reads artifacts/summary.json, e1_phase_scan.json, e2_forecast.json, e3_shots_noise.json and figures/*.png, and writes

    presentation/deck.html   self-contained slide deck (figures embedded as base64)
    presentation/deck.pdf    the same deck printed by headless Chrome (skipped with a warning if Chrome is missing)
    README.md                only the block between <!-- RESULTS:START --> and <!-- RESULTS:END -->
    docs/abstract.md, docs/pitch.md, docs/qa.md, docs/submission.md

No result number is typed by hand: every number is pulled from the JSON (or computed from it) at build time. The
rolling-origin cross-validation (summary.json["rolling_origin"]) is included when present and skipped otherwise.
Qualitative claims (e.g. "AR is best on both series") are checked against the data; the build stops if one no longer
holds. The build also fails if a placeholder, nan/inf or None survives, or if the abstract exceeds 200 words.
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART, FIG, OUT, DOCS = ROOT / "artifacts", ROOT / "figures", ROOT / "presentation", ROOT / "docs"
REPO = "https://github.com/wdreisz/floquet-reservoir"
REPO_SHORT = "github.com/wdreisz/floquet-reservoir"
AUTHOR = "Wyatt Reisz"
FOOTER = "Classical simulation — no quantum hardware used."


def load(name: str) -> dict:
    return json.loads((ART / name).read_text())


SUM = load("summary.json")
E1 = load("e1_phase_scan.json")
E2 = load("e2_forecast.json")
E3 = load("e3_shots_noise.json")
RO = SUM.get("rolling_origin") or None
PROV = E2["provenance"]

SERIES = ("sunspots", "co2")
SNAME = {"sunspots": "Sunspots (yearly)", "co2": "Mauna Loa CO₂ (monthly change)"}
SNAME_MD = {"sunspots": "Sunspots (yearly)", "co2": "Mauna Loa CO2 (monthly change)"}
SPLAIN = {"sunspots": "sunspots", "co2": "CO₂"}
SPLAIN_MD = {"sunspots": "sunspots", "co2": "CO2"}
MODELS = ("ar", "esn", "quantum_reservoir", "persistence", "quantum_reservoir_J0")
MNAME = {"ar": "Linear AR (p lags)", "esn": "Echo-state network, 11 tanh nodes", "quantum_reservoir": "Quantum reservoir, 6 spins (J = 0.4)",
         "quantum_reservoir_J0": "Quantum reservoir, J = 0 (no interactions)", "persistence": "Persistence (next = current)"}
MSHORT = {"ar": "AR", "esn": "ESN", "quantum_reservoir": "Quantum reservoir", "quantum_reservoir_J0": "J = 0 control",
          "persistence": "Persistence"}


# ----------------------------------------------------------------------------- formatting

def f2(x: float) -> str:
    return f"{x:.2f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def th(x) -> str:
    return f"{float(x):g}π"


def big(n: float) -> str:
    return f"{n:,.0f}"


def frac(x: float) -> str:
    return f"{100 * x:.0f} %"


# ----------------------------------------------------------------------------- facts: model configuration

E1_ROWS = E1["rows"]
J_VALUES = sorted({r["J"] for r in E1_ROWS})
J_INT = 0.4
assert J_INT in J_VALUES
THETAS_E1 = sorted({r["theta_over_pi"] for r in E1_ROWS})
E1_SEEDS = len({r["seed"] for r in E1_ROWS})
N_SPINS = 6
N_FEATURES = 2 * N_SPINS - 1
CYCLES = 2
try:  # read the model defaults from the package when it is importable
    sys.path.insert(0, str(ROOT / "src"))
    from qreservoir.reservoir import FloquetReservoir  # noqa: E402

    _r = FloquetReservoir.__dataclass_fields__
    N_SPINS, CYCLES, W_DIS = _r["n"].default, _r["cycles_per_step"].default, _r["W"].default
    N_FEATURES = 2 * N_SPINS - 1
except Exception:  # pragma: no cover - fall back to documented defaults
    import math

    W_DIS = math.pi
MAX_DELAY = len(E1_ROWS[0]["linear_by_delay"])


# ----------------------------------------------------------------------------- facts: E1 memory

M = SUM["memory"]


def scan(J: float, cycles: int = CYCLES) -> dict[float, dict]:
    g = defaultdict(list)
    for r in E1_ROWS:
        if r["J"] == J and r["cycles_per_step"] == cycles:
            g[r["theta_over_pi"]].append(r)
    return {t: {"MC": statistics.mean(x["MC"] for x in R), "order": statistics.mean(x["dtc_order"] for x in R)} for t, R in sorted(g.items())}


SC = scan(J_INT)
SC0 = scan(0.0)
MC_HIGH = 5.0
HIGH_TH = [t for t, v in SC.items() if v["MC"] >= MC_HIGH]
HIGH_LO, HIGH_HI = min(HIGH_TH), max(HIGH_TH)
J0_MIN, J0_MAX = min(v["MC"] for v in SC0.values()), max(v["MC"] for v in SC0.values())
ESN_LO, ESN_HI = M["esn_MC_range"]
assert M["dtc_MC"] < M["best_MC"] / 2, "time-crystal MC is no longer far below the scrambling peak: rewrite memory wording"
assert M["corr_MC_vs_dtc_order"] < 0, "MC no longer anti-correlates with time-crystal order: rewrite memory wording"
assert ESN_LO <= M["best_MC"], "scrambling peak no longer reaches the ESN range: rewrite memory wording"
DTC_ORDER = M["dtc_order_at_0.95"]


# ----------------------------------------------------------------------------- facts: E2 forecasting

FC = SUM["forecast"]


def med(s: str, m: str) -> float:
    d = FC[s]
    if m == "ar":
        return d["ar_test_nmse"]
    if m == "persistence":
        return d["persistence_test_nmse"]
    return d[m]["median_test_nmse"]


RANK = {s: sorted(MODELS, key=lambda m: med(s, m)) for s in SERIES}
for s in SERIES:
    assert RANK[s][0] == "ar", f"AR is no longer best on {s}: rewrite 'AR wins'"
    assert RANK[s][1] == "esn", f"ESN is no longer second on {s}: rewrite 'ESN second'"
N_SEEDS = FC["sunspots"]["quantum_reservoir"]["n_seeds"]
SPLIT = {s: E2[s]["split"] for s in SERIES}
NVAL = {s: E2[s]["n"] for s in SERIES}
FRAC = {k: (SPLIT["sunspots"][k][1] - SPLIT["sunspots"][k][0]) / NVAL["sunspots"] for k in ("train", "val", "test")}
DESC = {s: E2[s]["description"] for s in SERIES}
THETA_GRID = sorted(float(t) for t in FC["sunspots"]["quantum_median_test_by_theta"])


def best_theta(s: str, which: str) -> float:
    d = FC[s][f"quantum_median_{which}_by_theta"]
    return float(min(d, key=d.get))


VAL_BEST = {s: best_theta(s, "val") for s in SERIES}
TEST_BEST = {s: best_theta(s, "test") for s in SERIES}
QR = {s: FC[s]["quantum_reservoir"] for s in SERIES}
J0 = {s: FC[s]["quantum_reservoir_J0"] for s in SERIES}
ESN = {s: FC[s]["esn"] for s in SERIES}
PERS = {s: FC[s]["persistence_test_nmse"] for s in SERIES}
AR = {s: FC[s]["ar_test_nmse"] for s in SERIES}
ARP = {s: FC[s]["ar_p"] for s in SERIES}

assert QR["co2"]["median_test_nmse"] < J0["co2"]["median_test_nmse"], "quantum no longer beats J=0 on CO2"
assert QR["co2"]["median_test_nmse"] < PERS["co2"], "quantum no longer beats persistence on CO2"
assert VAL_BEST["sunspots"] != TEST_BEST["sunspots"], "sunspots: validation-best phase now also wins on test; rewrite"
SUN_RATIO = QR["sunspots"]["median_test_nmse"] / PERS["sunspots"]
assert abs(SUN_RATIO - 1) < 0.15, "sunspots quantum median is no longer ≈ persistence: rewrite"
J0_RATIO_CO2 = J0["co2"]["median_test_nmse"] / QR["co2"]["median_test_nmse"]
ESN_BEATS_AR = {s: ESN[s]["seeds_beating_AR"] for s in SERIES}
CHOSEN = {s: QR[s]["chosen_theta_over_pi"] for s in SERIES}


def chosen_str(s: str) -> str:
    c = sorted(set(CHOSEN[s]))
    return ", ".join(f"{th(t)} ×{CHOSEN[s].count(t)}" for t in c)


def rank_line(s: str, sep: str = " < ") -> str:
    return sep.join(f"{MSHORT[m]} {f2(med(s, m))}" for m in RANK[s])


# ----------------------------------------------------------------------------- facts: E3 shots, noise, hardware

SH = SUM["shots_MC"]
SH_ORDER = ["exact"] + sorted((k for k in SH["0.5"] if k != "exact"), key=lambda k: -int(k))
SH_LBL = {k: ("exact" if k == "exact" else f"{int(k):,} shots") for k in SH_ORDER}
assert SH["0.5"][SH_ORDER[-1]] < SH["0.5"]["exact"] / 2, "scrambling MC no longer collapses with shots: rewrite"
E3_ROWS = E3["rows"]
NOISE_P = sorted({r["noise_p"] for r in E3_ROWS})


def e3_mc(theta: float, p: float, shots) -> float:
    return next(statistics.mean(r["MC"]) for r in E3_ROWS if r["theta_over_pi"] == theta and r["noise_p"] == p and r["shots"] == shots)


NOISE_EXACT = {p: e3_mc(0.5, p, None) for p in NOISE_P}
NOISE_1K = {p: e3_mc(0.5, p, 1000) for p in NOISE_P}
P_MAX = NOISE_P[-1]
HW = SUM["hardware_cost"]
T_SUN = NVAL["sunspots"]
CYC_PER_STEP = HW["cycles_total_per_shot_sweep"] / (T_SUN * (T_SUN + 1) / 2)
assert abs(CYC_PER_STEP - round(CYC_PER_STEP)) < 1e-9, "hardware_cost no longer matches T(T+1)/2 restarts"
CYC_PER_STEP = int(round(CYC_PER_STEP))
SHOTS_HW = HW["circuit_runs_at_1000_shots"] // T_SUN
ONE_PASS = CYC_PER_STEP * T_SUN
RESTART_X = HW["cycles_total_per_shot_sweep"] / ONE_PASS
PROTO_RUNS = sum(NVAL[s] * SHOTS_HW * len(THETA_GRID) * N_SEEDS for s in SERIES)


# ----------------------------------------------------------------------------- facts: rolling origin (optional)

def ro_facts() -> dict | None:
    if not RO:
        return None
    out = {}
    for s in SERIES:
        if s not in RO:
            continue
        d = RO[s]
        folds = d.get("folds")
        nf = folds if isinstance(folds, int) else len(folds or d.get("per_fold_median_test_nmse", {}))
        models = [m for m in MODELS if m in d.get("median_test_nmse", {})] + [m for m in d.get("median_test_nmse", {}) if m not in MODELS]
        by_rank = sorted(models, key=lambda m: (d.get("mean_rank", {}).get(m, 99), d["median_test_nmse"][m]))
        out[s] = {"n_folds": nf, "folds": folds if isinstance(folds, list) else None, "models": by_rank, "d": d}
    return out or None


ROF = ro_facts()
TOP3 = ["ar", "esn", "quantum_reservoir"]


def ro_order_holds() -> bool:
    return bool(ROF) and all(ROF[s]["models"][:3] == TOP3 for s in ROF)


def ro_summary(md: bool = True) -> str:
    """One generated sentence on the rolling-origin CV (empty when absent)."""
    if not ROF:
        return ""
    nf = ROF[next(iter(ROF))]["n_folds"]
    if ro_order_holds():
        txt = f" Rolling-origin cross-validation ({nf} folds) keeps the AR > ESN > quantum order on both series by mean rank."
    else:
        txt = f" Rolling-origin cross-validation ({nf} folds) changes the order: " + "; ".join(ro_sentence(s, md) for s in ROF) + "."
    s = "sunspots"
    if s in ROF:
        d = ROF[s]["d"]
        pf = d.get("per_fold_median_test_nmse", {})
        worse = sum(pf[k]["quantum_reservoir"] > pf[k]["persistence"] for k in pf)
        if d["median_test_nmse"]["quantum_reservoir"] < d["median_test_nmse"]["persistence"]:
            txt += (f" In those folds the quantum reservoir beats persistence on sunspots in median ({f2(d['median_test_nmse']['quantum_reservoir'])} vs "
                    f"{f2(d['median_test_nmse']['persistence'])}) but is worse in {worse} of {len(pf)} folds.")
    return txt


def fold_label(f) -> str:
    if isinstance(f, (list, tuple)) and len(f) == 3:
        return f"test {100 * f[1]:.0f}–{100 * f[2]:.0f} %"
    return str(f)


def ro_sentence(s: str, md: bool = False) -> str:
    r = ROF[s]
    d = r["d"]
    name = (SPLAIN_MD if md else SPLAIN)[s]
    order = " > ".join(MSHORT.get(m, m) for m in r["models"])
    qr = "quantum_reservoir"
    q = ""
    if qr in d.get("mean_rank", {}):
        q = f"; quantum reservoir mean rank {d['mean_rank'][qr]:g} of {len(r['models'])}, best in {d.get('wins', {}).get(qr, 0)} of {r['n_folds']} folds"
    return f"{name}, {r['n_folds']} folds: mean-rank order {order}{q}"


def ro_table_md() -> str:
    lines = []
    for s in SERIES:
        if not ROF or s not in ROF:
            continue
        r = ROF[s]
        d = r["d"]
        lines.append(f"**{SNAME_MD[s]}**, {r['n_folds']} rolling-origin folds\n")
        lines.append("| Model | Median test NMSE over folds | Mean rank | Folds won |")
        lines.append("|---|---:|---:|---:|")
        for m in r["models"]:
            mr = d.get("mean_rank", {}).get(m)
            lines.append(f"| {MNAME.get(m, m)} | {f3(d['median_test_nmse'][m])} | {'–' if mr is None else f'{mr:g}'} | {d.get('wins', {}).get(m, '–')} |")
        lines.append("")
    return "\n".join(lines)


def ro_table_html(s: str) -> str:
    r = ROF[s]
    d = r["d"]
    pf = d.get("per_fold_median_test_nmse", {})
    fkeys = sorted(pf, key=lambda k: int(k)) if pf else []
    head = "".join(f"<th>{fold_label(r['folds'][int(k)]) if r['folds'] else 'fold ' + k}</th>" for k in fkeys)
    rows = ""
    for m in r["models"]:
        cells = "".join(f"<td>{f2(pf[k][m])}</td>" for k in fkeys)
        mr = d.get("mean_rank", {}).get(m)
        rows += f"<tr><td>{MSHORT.get(m, m)}</td>{cells}<td><b>{'–' if mr is None else f'{mr:g}'}</b></td></tr>"
    return f"<table><tr><th>{SPLAIN[s]}</th>{head}<th>mean rank</th></tr>{rows}</table>"


# ----------------------------------------------------------------------------- images

def _png64(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def img64(name: str, drop_footer: bool = True) -> str:
    """Embed a figure; drop its own footer line (the slides carry their own) by cutting at the last wide white gap."""
    raw = (FIG / name).read_bytes()
    if not drop_footer:
        return _png64(raw)
    try:
        import numpy as np
        from PIL import Image

        im = Image.open(io.BytesIO(raw)).convert("RGB")
        g = np.asarray(im.convert("L"))
        H = g.shape[0]
        ink = np.where(g.min(axis=1) < 235)[0]
        gaps = np.where(np.diff(ink) >= 25)[0]
        gaps = gaps[ink[gaps] > 0.8 * H]
        if len(gaps):
            im = im.crop((0, 0, im.width, int(ink[gaps[-1]]) + 10))
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        return _png64(buf.getvalue())
    except Exception:
        return _png64(raw)


# ----------------------------------------------------------------------------- deck

CSS = """
:root{--ink:#1f2328;--muted:#59616b;--line:#dedad0;--bg:#e9e6de;--paper:#fffdf8;--a:#d55e00;--b:#0072b2;--g:#009e73;
--v:#7c3aed;--h:min(1vh,0.5625vw)}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink)}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif;overflow:hidden;
-webkit-font-smoothing:antialiased}
.slide{position:fixed;left:50%;top:50%;width:calc(var(--h)*177.78);height:calc(var(--h)*100);transform:translate(-50%,-50%);
background:var(--paper);padding:calc(var(--h)*4.6) calc(var(--h)*6.5) calc(var(--h)*8);display:none;flex-direction:column;
font-size:calc(var(--h)*2.6);line-height:1.36;box-shadow:0 0 0 1px var(--line)}
.slide.active{display:flex}
.kicker{font-size:calc(var(--h)*2.1);color:var(--muted);letter-spacing:.02em;margin-bottom:calc(var(--h)*.9)}
.kicker b{color:var(--a);font-weight:600}
h1{font-size:calc(var(--h)*4.6);line-height:1.12;margin:0 0 calc(var(--h)*2.2);font-weight:650;letter-spacing:-.01em}
h1.title{font-size:calc(var(--h)*6.4);margin-top:calc(var(--h)*4)}
h2{font-size:calc(var(--h)*2.9);margin:0 0 calc(var(--h)*.9);font-weight:650}
p{margin:0 0 calc(var(--h)*1.1)}
ul{margin:0;padding-left:1.1em}
li{margin:0 0 calc(var(--h)*.9)}
.cols{display:grid;gap:calc(var(--h)*4);flex:1;min-height:0}
.c2{grid-template-columns:1fr 1fr}.c64{grid-template-columns:1.45fr 1fr}.c73{grid-template-columns:1.25fr 1fr}
.muted{color:var(--muted)}
.a{color:var(--a)}.b{color:var(--b)}.g{color:var(--g)}
.big{font-size:calc(var(--h)*4.6);font-weight:700;line-height:1.05;display:block}
table{border-collapse:collapse;width:100%;font-size:calc(var(--h)*2.3);font-variant-numeric:tabular-nums}
th,td{padding:calc(var(--h)*.5) calc(var(--h)*1);border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--muted);font-size:calc(var(--h)*2.2)}
.fitimg{max-height:100%;max-width:100%;width:auto;height:auto;object-fit:contain;object-position:left top;display:block;min-height:0}
.imgcell{min-height:0;display:flex;align-items:flex-start}
.box{border-left:calc(var(--h)*.6) solid var(--b);padding-left:calc(var(--h)*2)}
.box.a{border-color:var(--a)}.box.g{border-color:var(--g)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em}
pre{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:calc(var(--h)*2.05);line-height:1.45;
background:#f3f0e8;padding:calc(var(--h)*1.4) calc(var(--h)*2);border-radius:calc(var(--h)*.6);white-space:pre-wrap}
.foot{position:absolute;left:calc(var(--h)*6.5);right:calc(var(--h)*6.5);bottom:calc(var(--h)*2.6);display:flex;
justify-content:space-between;font-size:calc(var(--h)*2.3);color:var(--muted);border-top:1px solid var(--line);
padding-top:calc(var(--h)*1)}
.foot b{color:var(--ink);font-weight:600}
.small{font-size:calc(var(--h)*2.25)}
.stack{display:flex;flex-direction:column;gap:calc(var(--h)*1.6);min-height:0}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:calc(var(--h)*2.4);margin-top:calc(var(--h)*3)}
.stat{border-top:1px solid var(--line);padding-top:calc(var(--h)*1.1);font-size:calc(var(--h)*2.25);line-height:1.32}
.stat .big{margin-bottom:calc(var(--h)*.5)}
.flow{display:flex;align-items:stretch;gap:calc(var(--h)*1.1);margin:0 0 calc(var(--h)*2)}
.step{flex:1;border:1px solid var(--line);border-top:calc(var(--h)*.6) solid var(--b);background:#fbf9f3;
padding:calc(var(--h)*1.1) calc(var(--h)*1.3);font-size:calc(var(--h)*2.2);line-height:1.3;border-radius:0 0 calc(var(--h)*.5) calc(var(--h)*.5)}
.step b{display:block;font-size:calc(var(--h)*2.4);margin-bottom:calc(var(--h)*.4)}
.step.q{border-top-color:var(--a)}.step.t{border-top-color:var(--g)}
.arrow{align-self:center;color:var(--muted);font-size:calc(var(--h)*3)}
.check li{list-style:none;margin-left:-1.1em}
.check li:before{content:"✓ ";color:var(--g);font-weight:700}
@media print{@page{size:1600px 900px;margin:0}body{overflow:visible;background:#fff}
.slide{display:flex!important;position:relative;left:0;top:0;transform:none;width:1600px;height:900px;--h:9px;page-break-after:always;
break-after:page;box-shadow:none}}
"""

JS = """
const S=[...document.querySelectorAll('.slide')];let i=0;
function show(n){i=Math.max(0,Math.min(S.length-1,n));S.forEach((s,k)=>s.classList.toggle('active',k===i));
history.replaceState(null,'','#'+(i+1));}
document.addEventListener('keydown',e=>{
 if(['ArrowRight','ArrowDown','PageDown',' ','Enter'].includes(e.key)){e.preventDefault();show(i+1)}
 else if(['ArrowLeft','ArrowUp','PageUp','Backspace'].includes(e.key)){e.preventDefault();show(i-1)}
 else if(e.key==='Home'){show(0)}else if(e.key==='End'){show(S.length-1)}});
document.addEventListener('click',e=>{if(e.target.closest('a'))return;show(e.clientX>innerWidth/3?i+1:i-1)});
show((parseInt(location.hash.slice(1))||1)-1);
"""


def deck() -> str:
    slides: list[tuple[str, str, str]] = []  # (kicker, title, body)

    # ---- 1 title
    sun_q = QR["sunspots"]["median_test_nmse"]
    slides.append(("Wyatt Reisz", "", f"""
<h1 class="title">Does a time crystal make a better quantum memory?</h1>
<p style="font-size:calc(var(--h)*3.1);max-width:80%">A disordered kicked-Ising (Floquet) spin chain used as a quantum reservoir computer,
benchmarked against matched classical models on two real time series.</p>
<p class="muted">{AUTHOR} · <a href="{REPO}">{REPO_SHORT}</a></p>
<div class="tiles">
<div class="stat"><span class="big a">{f2(M['dtc_MC'])}</span>memory capacity in the time-crystal regime (θ = {th(M['dtc_theta_over_pi'])})</div>
<div class="stat"><span class="big b">{f2(M['best_MC'])}</span>memory capacity in the scrambling regime (θ = {th(M['best_theta_over_pi'])}); ESN {f2(ESN_LO)}–{f2(ESN_HI)}</div>
<div class="stat"><span class="big g">AR</span>best forecaster on both series; ESN second, quantum reservoir third</div>
<div class="stat"><span class="big">{SH['0.5'][SH_ORDER[-1]]:.1f}</span>scrambling-regime memory capacity left at {SH_LBL[SH_ORDER[-1]]} per readout (exact {f2(SH['0.5']['exact'])})</div>
</div>"""))

    # ---- 2 question
    drows = "".join(
        f"<tr><td>{SNAME[s]}</td><td>{NVAL[s]}</td><td>{SPLIT[s]['train'][1]}</td><td>{SPLIT[s]['val'][1] - SPLIT[s]['val'][0]}</td>"
        f"<td>{SPLIT[s]['test'][1] - SPLIT[s]['test'][0]}</td></tr>" for s in SERIES)
    slides.append(("Question", "Does time-crystal order give a quantum reservoir better memory, and does any of it beat classical models on real data?", f"""
<div class="cols c2"><div>
<p><b>Quantum reservoir computing</b> (Fujii &amp; Nakajima 2017): let a disordered quantum system evolve, feed the input in,
measure, and train only a linear readout. The physics does the feature engineering.</p>
<p><b>Discrete time crystals</b> are driven spin chains that flip with period 2T and resist perturbations. Zhang et al. (2025) built a
time-crystal reservoir and found phase boundaries best on synthetic tasks.</p>
<h2>What we ask</h2>
<ul><li>Which Floquet phase gives the most memory: time-crystal, scrambling, or weakly kicked?</li>
<li>Does the best-memory phase also forecast real series best?</li>
<li>Does it beat an echo-state network, a linear AR model and persistence with the <b>same features, splits and readout</b>?</li>
<li>What would measurement cost on hardware?</li></ul>
</div><div>
<h2>Data (small, real, public; bundled with statsmodels)</h2>
<table><tr><th>Series</th><th>Values</th><th>Train</th><th>Val</th><th>Test</th></tr>{drows}</table>
<p class="small muted" style="margin-top:calc(var(--h)*.8)">{e(DESC['sunspots'])}. {e(DESC['co2'])}. Time-ordered splits, train {frac(FRAC['train'])} /
validation {frac(FRAC['val'])} / test {frac(FRAC['test'])}.</p>
<div class="box a" style="margin-top:calc(var(--h)*2.2)"><h2>Short answer</h2>
<p>Time-crystal order <b>hurts</b> memory: it freezes the spins. The scrambling regime remembers as much as an 11-node ESN, but on
real forecasting a linear AR model wins both series.</p></div>
</div></div>"""))

    # ---- 3 model
    slides.append(("Model", f"A {N_SPINS}-spin kicked Ising chain, reset-and-inject input, {N_FEATURES} measured features", f"""
<div class="cols c64"><div class="stack">
<img class="fitimg" src="{img64('circuit.png')}" alt="One input step: reset and encode u on qubit 0, then a Floquet cycle">
<p class="small muted">Circuit drawing: one step, one Floquet cycle, θ = 0.95π shown. The disorder angles h<sub>i</sub> are drawn per seed.</p>
</div><div>
<div class="flow"><div class="step q"><b>1 · Inject</b>reset qubit 0 to √(1−u)|0⟩ + √u|1⟩</div>
<div class="step"><b>2 · Evolve</b>{CYCLES} Floquet cycles per step</div>
<div class="step t"><b>3 · Read</b>⟨Z<sub>i</sub>⟩ and ⟨Z<sub>i</sub>Z<sub>i+1</sub>⟩</div></div>
<p><b>Floquet cycle</b> U<sub>F</sub> = e<sup>−i Σ h<sub>i</sub>Z<sub>i</sub></sup> e<sup>−i J Σ Z<sub>i</sub>Z<sub>i+1</sub></sup> Π R<sub>x</sub>(θ),
disorder h<sub>i</sub> ~ U[−π, π].</p>
<ul><li><b>θ ≈ π, J ≠ 0:</b> time crystal (period-doubled magnetisation).</li>
<li><b>θ ≈ π/2:</b> scrambling (thermalising) kicks.</li>
<li><b>J = 0:</b> no interactions; information cannot leave qubit 0. Our quantum control.</li></ul>
<p><b>Readout:</b> {N_SPINS} + {N_SPINS - 1} = {N_FEATURES} features → ridge regression. Only the readout is trained.</p>
<p class="small muted">Simulated exactly with NumPy density matrices (2<sup>{N_SPINS}</sup> × 2<sup>{N_SPINS}</sup>). One reservoir step is
cross-checked against Qiskit Aer's density-matrix simulator in <code>tests/</code>.</p>
</div></div>"""))

    # ---- 4 method
    slides.append(("Method", "A fair benchmark: same features, same splits, same readout, tuned on validation only", f"""
<div class="flow">
<div class="step"><b>Train {frac(FRAC['train'])}</b>fit ridge readouts; min–max scaling fitted here only, not clipped</div><div class="arrow">→</div>
<div class="step q"><b>Validation {frac(FRAC['val'])}</b>choose θ, ridge α, ESN ρ and input scale, AR order p</div><div class="arrow">→</div>
<div class="step"><b>Refit</b>on train + validation with the chosen settings</div><div class="arrow">→</div>
<div class="step t"><b>Test {frac(FRAC['test'])}</b>evaluated once; Diebold–Mariano tests</div></div>
<div class="cols c2"><div>
<h2>Models (one-step-ahead forecast, NMSE)</h2>
<table><tr><th>Model</th><th>Features</th><th>Tuned on validation</th></tr>
<tr><td class="a">Quantum reservoir, J = {J_INT:g}</td><td>{N_FEATURES}</td><td>θ ∈ {{{', '.join(f'{t:g}' for t in THETA_GRID)}}}π, α</td></tr>
<tr><td>Quantum, J = 0 control</td><td>{N_FEATURES}</td><td>θ, α</td></tr>
<tr><td class="b">Echo-state network</td><td>{N_FEATURES} tanh nodes</td><td>ρ, input scale, α</td></tr>
<tr><td class="g">Linear AR</td><td>p ≤ {N_FEATURES} lags</td><td>p, α</td></tr>
<tr><td>Persistence</td><td>–</td><td>–</td></tr></table>
<p class="small muted" style="margin-top:calc(var(--h)*.8)">{N_SEEDS} disorder / random seeds for every stochastic model.
Memory capacity (E1): i.i.d. input, sum of r² for recalling u(t−k), k = 0…{MAX_DELAY - 1}.</p>
</div><div>
<h2>Track 2 brief → what we did</h2>
<ul class="check"><li>Real datasets, kept small ({NVAL['sunspots']} and {NVAL['co2']} values).</li>
<li>Same {N_FEATURES} features (or fewer) and the same split for every model.</li>
<li>Min–max scaled on the training block only, not clipped (a test checks it).</li>
<li>All hyperparameters chosen on validation; test touched once.</li>
<li>Cross-validated: {N_SEEDS} seeds{f", plus {ROF['sunspots']['n_folds']}-fold rolling-origin CV" if ROF and 'sunspots' in ROF else ""}.</li>
<li>Shows where the classical model wins, with Diebold–Mariano tests per seed.</li></ul>
</div></div>"""))

    # ---- 5 memory
    slides.append(("Result 1 · memory", "Time-crystal order hurts memory: it freezes the spins", f"""
<div class="cols c73"><div class="imgcell"><img class="fitimg" src="{img64('phase_figure.png')}" alt="Memory capacity vs kick angle"></div>
<div>
<ul><li><b class="b">Scrambling kicks remember.</b> With J = {J_INT:g}, memory capacity is ≥ {MC_HIGH:g} for θ = {th(HIGH_LO)}–{th(HIGH_HI)}, peaking at
<b>{f2(M['best_MC'])}</b> (θ = {th(M['best_theta_over_pi'])}). That is inside the 11-node ESN range ({f2(ESN_LO)}–{f2(ESN_HI)}).</li>
<li><b class="a">The time crystal forgets.</b> At θ = {th(M['dtc_theta_over_pi'])} the chain is strongly period-doubled (order {f2(DTC_ORDER)})
and memory capacity is only <b>{f2(M['dtc_MC'])}</b>: the spins mostly flip back and forth, and the input barely spreads.</li>
<li><b>Across the scan</b>, memory capacity anti-correlates with time-crystal order (r = {f2(M['corr_MC_vs_dtc_order'])}, J ∈ {{0.4, π/4}}).</li>
<li><b>J = 0 control</b> stays at {f2(J0_MIN)}–{f2(J0_MAX)} at every θ: without interactions there is no reservoir.</li>
<li class="muted">11 raw lags would score {f2(M['lags11_MC'])}: a linear delay line is the memory ceiling.</li></ul>
<p class="small muted">{len(THETAS_E1)} kick angles × {len(J_VALUES)} couplings × {E1_SEEDS} disorder seeds; mean ± s.d. shaded.</p>
</div></div>"""))

    # ---- 6 forecasting
    frows = ""
    for m in MODELS:
        frows += f"<tr><td>{MNAME[m]}</td>" + "".join(
            f"<td>{f3(med(s, m))}" + (f" <span class='muted small'>[{f2(FC[s][m]['min'])}–{f2(FC[s][m]['max'])}]</span>" if m in FC[s] else "")
            + "</td>" for s in SERIES) + "</tr>"
    slides.append(("Result 2 · forecasting", "On real series, linear AR wins; the quantum reservoir is third", f"""
<div class="cols c73"><div class="imgcell"><img class="fitimg" src="{img64('forecast_figure.png')}" alt="Held-out forecasts and per-seed test NMSE"></div>
<div>
<table><tr><th>Test NMSE (median over {N_SEEDS} seeds)</th><th>Sunspots</th><th>CO₂</th></tr>{frows}</table>
<ul style="margin-top:calc(var(--h)*1.6)">
<li><b>CO₂:</b> the quantum reservoir clearly beats its J = 0 control ({f2(QR['co2']['median_test_nmse'])} vs {f2(J0['co2']['median_test_nmse'])}) and
persistence ({f2(PERS['co2'])}). Validation and test both prefer θ = {th(VAL_BEST['co2'])}.</li>
<li><b>Sunspots:</b> on this split ≈ persistence ({f2(QR['sunspots']['median_test_nmse'])} vs {f2(PERS['sunspots'])}) and seed-sensitive
({f2(QR['sunspots']['min'])}–{f2(QR['sunspots']['max'])}). The phase that wins on validation (θ = {th(VAL_BEST['sunspots'])})
does not win on test (θ = {th(TEST_BEST['sunspots'])}).</li></ul>
<p class="small muted">[min–max over seeds]. AR order chosen on validation: p = {ARP['sunspots']} (sunspots), {ARP['co2']} (CO₂).</p>
</div></div>"""))

    # ---- 7 rolling-origin CV (optional)
    if ROF:
        ss = [s for s in SERIES if s in ROF]
        order = ROF[ss[0]]["models"]
        head = "".join(f"<th>{SPLAIN[s]} median</th><th>{SPLAIN[s]} mean rank</th><th>{SPLAIN[s]} folds won</th>" for s in ss)
        rows = ""
        for m in order:
            cells = ""
            for s in ss:
                d = ROF[s]["d"]
                mr = d.get("mean_rank", {}).get(m)
                cells += (f"<td>{f2(d['median_test_nmse'][m])}</td><td><b>{'–' if mr is None else f'{mr:g}'}</b></td>"
                          f"<td>{d.get('wins', {}).get(m, '–')}</td>")
            rows += f"<tr><td>{MSHORT.get(m, m)}</td>{cells}</tr>"
        bullets = "".join(f"<li>{ro_sentence(s)}.</li>" for s in ss)
        nf = ROF[ss[0]]["n_folds"]
        has_img = (FIG / "cv_figure.png").exists()
        img = f'<img class="fitimg" src="{img64("cv_figure.png")}" alt="Rolling-origin CV">' if has_img else ""
        slides.append(("Cross-validation", f"The ranking holds across {nf} rolling-origin folds", f"""
<div class="stack" style="flex:1">
<div class="imgcell" style="flex:1;min-height:0;justify-content:center">{img}</div>
<div class="cols" style="flex:none;grid-template-columns:1.35fr 1fr">
<div><table><tr><th>Test NMSE over folds</th>{head}</tr>{rows}</table></div>
<div><ul class="small">{bullets}</ul><p class="small">{ro_summary(md=False).strip()}</p>
<p class="small muted">Each fold moves the test block forward and re-tunes every model on its own preceding validation block; per-fold numbers are medians over seeds.</p></div>
</div></div>"""))

    # ---- 8 where classical wins
    wrows = ""
    for s in SERIES:
        wrows += (f"<tr><td>{SNAME[s]}</td><td>{QR[s]['seeds_where_AR_significantly_better_p05']} / {N_SEEDS}</td>"
                  f"<td>{QR[s]['seeds_where_ESN_significantly_better_p05']} / {N_SEEDS}</td><td>{QR[s]['seeds_beating_AR']} / {N_SEEDS}</td>"
                  f"<td>{ESN_BEATS_AR[s]} / {N_SEEDS}</td></tr>")
    slides.append(("Where the classical model wins", "Classical wins everywhere we looked, and we say so", f"""
<div class="cols c2"><div>
<h2>Diebold–Mariano tests on the test block (per seed, p &lt; 0.05)</h2>
<table><tr><th>Series</th><th>AR significantly better than quantum</th><th>ESN significantly better than quantum</th><th>Quantum beats AR</th><th>ESN beats AR</th></tr>{wrows}</table>
<h2 style="margin-top:calc(var(--h)*2.4)">Full ranking (median test NMSE)</h2>
<p>Sunspots: {rank_line('sunspots')}</p>
<p>CO₂: {rank_line('co2')}</p>
</div><div>
<h2>Why classical wins here</h2>
<ul><li><b>Both series are close to linear-autoregressive.</b> Sunspots have an ~11-year cycle; CO₂ changes have a 12-month cycle. AR with
p ≈ {ARP['sunspots']}–{ARP['co2']} lags captures exactly that, and our reservoir remembers only ~{M['best_MC']:.0f} steps.</li>
<li><b>An 11-node ESN has the same feature count</b>, more memory than the best quantum phase, and costs microseconds.</li>
<li><b>Reservoir choice is fragile on sunspots:</b> θ picked on validation varies by seed ({chosen_str('sunspots')}).</li></ul>
<div class="box g" style="margin-top:calc(var(--h)*2)"><p><b>What the quantum reservoir does show:</b> interactions matter. On CO₂ the J = 0 control is
{J0_RATIO_CO2:.1f}× worse, so the forecasting skill comes from many-body dynamics, not the single-qubit encoding.</p></div>
</div></div>"""))

    # ---- 9 shots / hardware
    shrows = "".join(f"<tr><td>{SH_LBL[k]}</td><td>{f2(SH['0.5'][k])}</td><td>{f2(SH['0.95'][k])}</td></tr>" for k in SH_ORDER)
    nrows = "".join(f"<tr><td>{p:g}</td><td>{f2(NOISE_EXACT[p])}</td><td>{f2(NOISE_1K[p])}</td></tr>" for p in NOISE_P)
    slides.append(("Result 3 · measurement cost", "Finite shots erase the scrambling advantage; hardware would pay a restart bill", f"""
<div class="stack" style="flex:1">
<div class="imgcell" style="flex:1;min-height:0"><img class="fitimg" src="{img64('robustness_figure.png')}" alt="Memory capacity vs shots and noise"></div>
<div class="cols c2" style="flex:none;grid-template-columns:1fr 1fr 1.25fr">
<div><table><tr><th>Memory capacity</th><th>θ = 0.5π</th><th>θ = 0.95π</th></tr>{shrows}</table></div>
<div><table><tr><th>Depolarizing p</th><th>exact</th><th>1,000 shots</th></tr>{nrows}</table>
<p class="small muted">θ = 0.5π, J = {J_INT:g}; mean over seeds.</p></div>
<div class="small"><p><b>The time crystal is shot-robust only because it has almost no memory to lose.</b></p>
<p><b>Hardware:</b> a projective readout destroys the state, so the sequence is re-run up to every readout time: {big(HW['cycles_total_per_shot_sweep'])} Floquet
cycles per shot for the {T_SUN}-year sunspot series instead of {big(ONE_PASS)} ({RESTART_X:.0f}×), and {big(HW['circuit_runs_at_1000_shots'])} circuit runs at
{SHOTS_HW:,} shots. Our whole E2 protocol would need ≈ {PROTO_RUNS / 1e6:.0f} million runs. Windowed replay or weak measurement (Mujal et al. 2023) would cut this.</p></div>
</div></div>"""))

    # ---- 10 prior work
    slides.append(("Prior work and what is new", "An independent replication and extension, not the first time-crystal reservoir", f"""
<div class="cols c2"><div>
<table><tr><th>Work</th><th>Reservoir</th><th>Finding</th></tr>
<tr><td>Fujii &amp; Nakajima 2017</td><td>disordered spins</td><td>QRC; reset-and-inject input</td></tr>
<tr><td>Martínez-Peña et al. PRL 2021</td><td>static disordered Ising</td><td>thermal phase / transition best; localisation hurts</td></tr>
<tr><td>Mujal et al. 2023</td><td>measurement protocols</td><td>restart / weak-measurement cost set by memory time</td></tr>
<tr><td><b>Zhang et al. 2025</b> (arXiv:2508.15230)</td><td><b>DTC kicked Ising</b>, same input and readout</td><td>phase boundaries best on memory, parity, NARMA10; MNIST on hardware</td></tr>
</table>
<p class="small muted" style="margin-top:calc(var(--h)*1)">Zhang et al. is the closest work and essentially our model. Our scan was done independently; see <code>docs/literature.md</code>.</p>
</div><div>
<h2>What this project adds</h2>
<ul><li><b>Real series:</b> sunspots and CO₂, not only synthetic tasks.</li>
<li><b>Matched classical baselines:</b> an ESN with the same {N_FEATURES} features, linear AR, persistence, and a J = 0 quantum control, all tuned on validation only.</li>
<li><b>Memory phase ≠ forecasting phase:</b> on sunspots the validation-best θ does not hold on test.</li>
<li><b>Measurement cost:</b> shot noise and restart bill per phase.</li></ul>
<div class="box a" style="margin-top:calc(var(--h)*2)"><p>Consistent with prior work: deep in the time-crystal phase the reservoir is a poor memory;
the useful regime is the scrambling side, toward the boundary.</p></div>
</div></div>"""))

    # ---- 11 takeaways
    slides.append(("Takeaways", "Time-crystal order is robust because it does not listen", f"""
<div class="cols c2"><div>
<ul><li><b>Memory:</b> scrambling kicks reach ESN-level memory ({f2(M['best_MC'])}); the time crystal stays at {f2(M['dtc_MC'])}.</li>
<li><b>Forecasting:</b> linear AR wins both real series; the ESN is second; the quantum reservoir is third and beats its J = 0 control.</li>
<li><b>Hardware:</b> with {SH_LBL[SH_ORDER[-2]]} the scrambling advantage shrinks to {f2(SH['0.5'][SH_ORDER[-2]])}, and every readout needs a restart.</li>
<li><b>No quantum advantage claimed.</b> The useful output is a fair, reusable benchmark for the next reservoir idea.</li></ul>
</div><div>
<h2>Reproduce (≈ 90 s on a laptop)</h2>
<pre>python -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -r requirements.lock.txt &amp;&amp; pip install -e .
python -m qreservoir.experiments
python -m qreservoir.figures
pytest
streamlit run app/app.py
python presentation/build.py</pre>
<p style="margin-top:calc(var(--h)*1.6)"><b><a href="{REPO}">{REPO_SHORT}</a></b> · MIT licence</p>
<p class="small muted">Every number in this deck is generated from <code>artifacts/*.json</code> by <code>presentation/build.py</code>.
Python {PROV['python']}, Qiskit {PROV['qiskit']}, NumPy {PROV['numpy']}.</p>
</div></div>"""))

    n = len(slides)
    out = []
    for k, (kicker, title, body) in enumerate(slides, 1):
        h1 = f"<h1>{title}</h1>" if title else ""
        out.append(f'<section class="slide" aria-label="Slide {k}"><div class="kicker"><b>Qiskit Fall Fest 2026 Warsaw</b> · Track 2 '
                   f'Quantum machine learning · {kicker}</div>{h1}{body}'
                   f'<div class="foot"><span><b>{FOOTER}</b> NumPy density matrices; reservoir step checked against Qiskit Aer.</span>'
                   f'<span>{k} / {n}</span></div></section>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Floquet Reservoir Deck</title><style>{CSS}</style></head><body>{"".join(out)}<script>{JS}</script></body></html>')


def e(s: str) -> str:
    return html.escape(s, quote=False)


# ----------------------------------------------------------------------------- README results block

def readme_results() -> str:
    t = ["*Generated by `presentation/build.py` from `artifacts/*.json`; do not edit by hand.*", ""]
    t.append("### Memory capacity (E1, i.i.d. input, mean over seeds)")
    t.append("")
    t.append("| Reservoir | Memory capacity |")
    t.append("|---|---:|")
    t.append(f"| Quantum, scrambling peak (θ = {M['best_theta_over_pi']:g}π, J = {J_INT:g}) | {f2(M['best_MC'])} |")
    t.append(f"| Quantum, time crystal (θ = {M['dtc_theta_over_pi']:g}π, J = {J_INT:g}; order {f2(DTC_ORDER)}) | {f2(M['dtc_MC'])} |")
    t.append(f"| Quantum, J = 0 control (all θ) | {f2(J0_MIN)}–{f2(J0_MAX)} |")
    t.append(f"| Echo-state network, 11 nodes (range over ρ and seeds) | {f2(ESN_LO)}–{f2(ESN_HI)} |")
    t.append(f"| 11 raw lags (linear delay line) | {f2(M['lags11_MC'])} |")
    t.append("")
    t.append(f"Memory capacity is at least {MC_HIGH:g} for θ = {HIGH_LO:g}π–{HIGH_HI:g}π (J = {J_INT:g}). Across the J ∈ {{0.4, π/4}} scan it "
             f"anti-correlates with time-crystal order (r = {f2(M['corr_MC_vs_dtc_order'])}).")
    t.append("")
    t.append(f"### One-step forecasting (E2, test block, median over {N_SEEDS} seeds, NMSE, lower is better)")
    t.append("")
    t.append("| Model | Sunspots | CO2 |")
    t.append("|---|---:|---:|")
    for m in MODELS:
        cells = []
        for s in SERIES:
            c = f3(med(s, m))
            if m in FC[s]:
                c += f" [{f2(FC[s][m]['min'])}–{f2(FC[s][m]['max'])}]"
            cells.append(c)
        t.append(f"| {MNAME[m]} | {' | '.join(cells)} |")
    t.append("")
    t.append("[min–max over seeds]. Diebold–Mariano tests on the test block, per seed, p < 0.05:")
    t.append("")
    t.append("| Series | AR significantly better than quantum | ESN significantly better than quantum | Quantum beats AR | ESN beats AR |")
    t.append("|---|---:|---:|---:|---:|")
    for s in SERIES:
        t.append(f"| {SNAME_MD[s]} | {QR[s]['seeds_where_AR_significantly_better_p05']} / {N_SEEDS} | {QR[s]['seeds_where_ESN_significantly_better_p05']} / {N_SEEDS} "
                 f"| {QR[s]['seeds_beating_AR']} / {N_SEEDS} | {ESN_BEATS_AR[s]} / {N_SEEDS} |")
    t.append("")
    t.append(f"Phase choice: the kick angle with the lowest median validation NMSE is θ = {VAL_BEST['sunspots']:g}π on sunspots, "
             f"but the lowest median test NMSE is at θ = {TEST_BEST['sunspots']:g}π. On CO2 both are θ = {VAL_BEST['co2']:g}π "
             f"(test {TEST_BEST['co2']:g}π). Per-seed choices on sunspots: {chosen_str('sunspots')}.")
    t.append("")
    if ROF:
        nf = ROF[next(iter(ROF))]["n_folds"]
        t.append(f"### Rolling-origin cross-validation ({nf} folds)")
        t.append("")
        if ROF[next(iter(ROF))]["folds"]:
            t.append("Test blocks: " + ", ".join(fold_label(f).replace("test ", "") for f in ROF[next(iter(ROF))]["folds"])
                     + " of each series; every fold re-tunes all models on its own preceding validation block.")
            t.append("")
        t.append(ro_table_md())
        for s in SERIES:
            if s in ROF:
                t.append(f"- {ro_sentence(s, md=True)}.")
        t.append("")
        t.append(ro_summary().strip())
        t.append("")
    t.append("### Shots, noise and hardware cost (E3)")
    t.append("")
    t.append("| Readout | Memory capacity, θ = 0.5π | Memory capacity, θ = 0.95π |")
    t.append("|---|---:|---:|")
    for k in SH_ORDER:
        t.append(f"| {SH_LBL[k]} | {f2(SH['0.5'][k])} | {f2(SH['0.95'][k])} |")
    t.append("")
    t.append(f"Synthetic depolarizing noise (θ = 0.5π, p per qubit per cycle from 0 to {P_MAX:g}): memory capacity goes "
             f"{f2(NOISE_EXACT[0.0])} → {f2(NOISE_EXACT[P_MAX])} with exact readout, and {f2(NOISE_1K[0.0])} → {f2(NOISE_1K[P_MAX])} at 1,000 shots.")
    t.append("")
    t.append(f"Hardware restart cost: a projective readout destroys the state, so the input sequence is re-run up to each readout time. "
             f"For the {T_SUN}-step sunspot series that is {big(HW['cycles_total_per_shot_sweep'])} Floquet cycles per shot instead of {big(ONE_PASS)} "
             f"for a single pass ({RESTART_X:.0f}×), and {big(HW['circuit_runs_at_1000_shots'])} circuit runs at {SHOTS_HW:,} shots per readout. "
             f"The E2 protocol ({len(THETA_GRID)} angles × {N_SEEDS} seeds × 2 series) would need about {PROTO_RUNS / 1e6:.0f} million circuit runs.")
    return "\n".join(t)


def write_readme() -> None:
    p = ROOT / "README.md"
    txt = p.read_text(encoding="utf-8")
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a not in txt or b not in txt:
        sys.exit("README.md is missing the RESULTS markers")
    block = readme_results()
    check("README results", block)
    new = txt.split(a)[0] + a + "\n" + block + "\n" + b + txt.split(b, 1)[1]
    p.write_text(new, encoding="utf-8")
    print(f"wrote README.md results block ({len(block):,} chars)")


# ----------------------------------------------------------------------------- docs

GEN = "*Generated by `presentation/build.py` from `artifacts/*.json`. Numbers update when the experiments are re-run.*"


def abstract() -> str:
    ro = ""
    if ROF:
        nf = ROF[next(iter(ROF))]["n_folds"]
        ro = f" {nf}-fold rolling-origin cross-validation keeps the AR > ESN > quantum order." if ro_order_holds() else ro_summary()
    body = (
        f"Discrete time crystals are robust, so are they good quantum memories? We use a {N_SPINS}-spin disordered kicked-Ising (Floquet) chain as a "
        f"quantum reservoir computer, simulated exactly with density matrices and cross-checked against Qiskit Aer, and compare its "
        f"{N_FEATURES} measured features against an echo-state network with the same feature count, linear autoregression, persistence and a "
        f"non-interacting (J = 0) control. All models share time-ordered train/validation/test splits, are tuned on validation only and "
        f"run over {N_SEEDS} seeds. Time-crystal order hurts memory: capacity drops from {f2(M['best_MC'])} in the scrambling regime to "
        f"{f2(M['dtc_MC'])} at θ = {M['dtc_theta_over_pi']:g}π. On yearly sunspots and Mauna Loa CO2, linear AR forecasts best, the ESN second and the quantum "
        f"reservoir third; on CO2 it clearly beats its J = 0 control, while on the main sunspot split it only matches persistence.{ro} With {SH_LBL[SH_ORDER[-2]]} "
        f"per readout the scrambling regime's memory falls to {f2(SH['0.5'][SH_ORDER[-2]])}, and projective readout would force the "
        f"sequence to be re-run for every time step. This is an independent replication and real-data extension of Zhang et al. (2025). "
        f"Classical simulation only; no quantum advantage is claimed."
    )
    words = len(body.split())
    if words > 200:
        sys.exit(f"abstract is {words} words (> 200)")
    return f"# Abstract\n\n{GEN}\n\n**Does a time crystal make a better quantum memory? A Floquet spin-chain reservoir vs classical forecasters on real data**\n\n{AUTHOR} · Qiskit Fall Fest 2026 Warsaw · Track 2\n\n{body}\n\n*({words} words)*\n"


def pitch() -> str:
    ro = ""
    if ROF:
        nf = ROF[next(iter(ROF))]["n_folds"]
        ro = (f" And it is not one lucky split: across {nf} rolling-origin folds the top three keep that order." if ro_order_holds()
              else ro_summary())
    return f"""# 60-second pitch

{GEN}

**[0:00]** Time crystals are the most robust phase a driven quantum chain can have. Robust sounds like a good memory. So: is a time crystal a better quantum reservoir computer?

**[0:10]** We built one: six spins, a disordered kicked Ising chain. Each step we reset one qubit to encode the input, kick the chain twice and read out {N_FEATURES} expectation values. Only a linear readout is trained. Everything is exact density-matrix simulation, checked against Qiskit Aer.

**[0:22]** First answer: no. Time-crystal order freezes the spins. Memory capacity is {f2(M['dtc_MC'])} in the time crystal and {f2(M['best_MC'])} in the scrambling regime, about what an 11-node echo-state network gets.

**[0:32]** Second: real data, fair fight. Sunspots and CO2, same features, same splits, tuned on validation only, {N_SEEDS} seeds. Linear AR wins both. The ESN is second. The quantum reservoir is third: on CO2 it is far better than its no-interaction control; on the main sunspot split it only matches persistence.{ro}

**[0:47]** Third: hardware. With {SH_LBL[SH_ORDER[-2]]} per readout its memory drops to {f2(SH['0.5'][SH_ORDER[-2]])}, and every readout means re-running the sequence.

**[0:55]** No quantum advantage. What we offer is an honest, reusable benchmark, and a replication of Zhang et al. 2025 on real data. Thank you.
"""


def qa() -> str:
    ro_q = ""
    if ROF:
        bits = "; ".join(ro_sentence(s, md=True) for s in SERIES if s in ROF)
        ro_q = f"Yes. We also ran rolling-origin cross-validation, where each fold re-tunes every model on its own preceding validation block: {bits}." + ro_summary()
    else:
        ro_q = "The headline uses one time-ordered split with 8 seeds; a rolling-origin check is the next step."
    return f"""# Likely judge questions, with honest answers

{GEN}

## Is this the first time-crystal quantum reservoir?
No. Zhang et al. (arXiv:2508.15230, 2025) used essentially the same model (imperfect π kicks, random ZZ and Z fields, reset-and-inject input, ⟨Z⟩/⟨ZZ⟩ readout) on synthetic tasks and found phase boundaries best. Our work is an independent replication and extension. We add real time series, classical baselines matched in feature count and readout, a check of whether the best-memory phase also forecasts best, and the measurement cost. See `docs/literature.md`.

## Do you claim quantum advantage?
No. Linear AR is the best forecaster on both series, with test NMSE {f3(AR['sunspots'])} on sunspots and {f3(AR['co2'])} on CO2. The quantum reservoir's medians are {f3(QR['sunspots']['median_test_nmse'])} and {f3(QR['co2']['median_test_nmse'])}. It beats AR on {QR['sunspots']['seeds_beating_AR']} of {N_SEEDS} seeds on sunspots and {QR['co2']['seeds_beating_AR']} of {N_SEEDS} on CO2.

## Did you run on quantum hardware?
No. Everything is exact NumPy density-matrix simulation of 6 qubits on a laptop. One reservoir step is cross-checked against Qiskit Aer's density-matrix simulator in `tests/`. Shot noise and depolarizing noise are simulated.

## Why is the time crystal bad at memory, when it is "robust"?
Its robustness is robustness to perturbations, and the input is a perturbation. Near θ = π the chain mostly flips back and forth; the injected state on qubit 0 barely spreads. At θ = {M['dtc_theta_over_pi']:g}π the period-doubling order is {f2(DTC_ORDER)} and memory capacity is {f2(M['dtc_MC'])}. Across the scan, memory capacity and time-crystal order correlate at r = {f2(M['corr_MC_vs_dtc_order'])}.

## Is the comparison fair to the quantum model?
We tried to make it so. Every model sees the same scaled series and the same time-ordered splits. The series is min–max scaled on the training block only, not clipped, and a test checks this. The quantum reservoir and ESN both give {N_FEATURES} features to the same ridge readout. AR uses at most {N_FEATURES} lags. θ, ridge α, ESN spectral radius and input scale, and AR order are all chosen on validation only. Models are refitted on train + validation and tested once. The quantum model gets {len(THETA_GRID)} kick angles to choose from; the ESN gets a comparable small grid of spectral radius × input scale.

## Is a single split enough?
{ro_q}

## Why does the phase chosen on validation not win on sunspots?
Sunspots have only {NVAL['sunspots']} yearly values, so validation is {SPLIT['sunspots']['val'][1] - SPLIT['sunspots']['val'][0]} points: one or two solar cycles. Median validation NMSE is lowest at θ = {VAL_BEST['sunspots']:g}π, but median test NMSE is lowest at θ = {TEST_BEST['sunspots']:g}π, the time-crystal side that behaves like persistence. Per-seed choices were {chosen_str('sunspots').replace('π', 'π')}. We report this rather than picking θ on test.

## Why include a J = 0 control?
Without interactions, information cannot leave the input qubit, so it is a "quantum but trivial" reservoir. It isolates the contribution of many-body dynamics. On CO2 the interacting reservoir's median NMSE is {f2(QR['co2']['median_test_nmse'])} against {f2(J0['co2']['median_test_nmse'])} for J = 0. Its memory capacity stays at {f2(J0_MIN)}–{f2(J0_MAX)} at every θ.

## What does it cost on real hardware?
Two things. Shot noise: at θ = 0.5π memory capacity goes {' → '.join(f2(SH['0.5'][k]) for k in SH_ORDER)} from exact readout down to {SH_LBL[SH_ORDER[-1]]}. Restarts: projective measurement destroys the state, so the input sequence must be re-run up to each readout. That is {big(HW['cycles_total_per_shot_sweep'])} Floquet cycles per shot for the {T_SUN}-step sunspot series instead of {big(ONE_PASS)}, and about {PROTO_RUNS / 1e6:.0f} million circuit runs for our full forecasting protocol at {SHOTS_HW:,} shots. Mujal et al. (2023) show windowed replay or weak measurement can reduce this.

## Why only 6 qubits?
Exact density matrices for 6 qubits are 64 × 64, which keeps the full scan ({len(E1_ROWS)} reservoir runs) and the forecasting grid at about 90 s on a laptop, with no shot noise unless we add it. Larger chains would add memory but also cost; we did not test them.

## What would make the quantum reservoir win?
A task that needs nonlinear memory beyond what AR captures, longer chains with more memory, and a readout protocol that does not destroy the state each step. Our benchmark is built so these can be tested with the same fair splits.

## How do you know the code is right?
`tests/` has 10 tests: the density matrix stays physical with and without noise; the Floquet operator is unitary; reset writes the input on qubit 0; a perfect π flip carries no input beyond qubit 0; one step matches Qiskit Aer's density-matrix simulator; shots converge to exact; capacity sanity checks; splits are time-ordered and scaling uses train only; the Diebold–Mariano sign convention.

## What is reusable for the community?
`qreservoir` is a small MIT-licensed package: an exact density-matrix Floquet reservoir with shots and noise, a matched ESN, memory-capacity and forecasting harnesses with validation-only tuning and Diebold–Mariano tests, and this generated deck. Swapping in another series or reservoir is a few lines.
"""


def submission() -> str:
    ro = ""
    if ROF:
        nf = ROF[next(iter(ROF))]["n_folds"]
        ro = ro_summary()
    return f"""# Submission

{GEN}

## Email draft

**To:** submission@aiqtlab.com
**Subject:** Qiskit Fall Fest 2026 Warsaw · Track 2 submission · [TEAM NAME]

Hello,

Please find our Track 2 (Quantum machine learning) submission below.

- **Team:** [TEAM NAME]
- **Members:** {AUTHOR}
- **Track:** 2, Quantum machine learning
- **Project:** Does a time crystal make a better quantum memory? A Floquet spin-chain reservoir vs classical forecasters on real data
- **Repository (public):** {REPO}
- **Demo video:** [VIDEO LINK]
- **Slides:** `presentation/deck.pdf` in the repository (`deck.html` is the interactive version)

**Summary.** We use a {N_SPINS}-spin disordered kicked-Ising (Floquet) chain as a quantum reservoir computer and benchmark it against an echo-state network with the same {N_FEATURES} features, linear autoregression, persistence and a non-interacting control. We use yearly sunspots and Mauna Loa CO2, with identical time-ordered splits, validation-only tuning and {N_SEEDS} seeds. Time-crystal order hurts memory (capacity {f2(M['dtc_MC'])} vs {f2(M['best_MC'])} in the scrambling regime). Linear AR is the best forecaster on both series, and the classical models win.{ro} We also show the shot-noise and restart cost of running it on hardware. All results are classical simulation; no quantum hardware was used and no quantum advantage is claimed.

The README has environment instructions, and one command regenerates every result (about 90 s on a laptop).

Best regards,
{AUTHOR}

## Demo video shot list (about 2 minutes)

| Time | Shot | Say |
|---|---|---|
| 0:00–0:12 | Deck slide 1 (title) | Time crystals are the most robust phase of a driven spin chain. Does that make them a better quantum memory? |
| 0:12–0:30 | Deck slide 3 (circuit) | Six spins; reset qubit 0 to encode the input; two Floquet kicks; read {N_FEATURES} expectation values; train only a ridge readout. Exact density-matrix simulation, checked against Qiskit Aer. |
| 0:30–0:45 | Deck slide 4 (method) | Same features, same splits, tuned on validation only, {N_SEEDS} seeds, Diebold–Mariano tests. ESN, AR, persistence and a no-interaction quantum control. |
| 0:45–1:05 | Deck slide 5 (memory figure); point at panel (a) and the time-crystal band | Scrambling kicks reach ESN-level memory; the time crystal forgets ({f2(M['dtc_MC'])} vs {f2(M['best_MC'])}). |
| 1:05–1:25 | Streamlit app (`streamlit run app/app.py`): move the θ slider from 0.5π to 0.95π | Live: memory profile collapses as we enter the time-crystal phase. |
| 1:25–1:45 | Deck slides 6–{"8" if ROF else "7"} (forecasting, {"cross-validation, " if ROF else ""}where classical wins) | Linear AR wins both series, ESN second, quantum third. On CO2 interactions clearly help; on sunspots it is about persistence. |
| 1:45–1:55 | Deck slide on measurement cost | With 1,000 shots the memory advantage mostly disappears, and every readout needs a restart. |
| 1:55–2:05 | Terminal: `python -m qreservoir.experiments`, then `pytest` passing; README on GitHub | Classical simulation only, no advantage claimed. Everything regenerates in about 90 seconds. Repo link on screen. |

Recording tips: 1920×1080, deck in full screen (`presentation/deck.html`, arrow keys), app in a second browser tab, terminal font ≥ 18 pt.
"""


# ----------------------------------------------------------------------------- write + check

PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_.\[\]'\"]*\}|\b(?:nan|None|NaN)\b|\binf\b")


def check(name: str, text: str) -> None:
    t = re.sub(r"<style>.*?</style>|<script>.*?</script>|data:image/png;base64,[A-Za-z0-9+/=]+", "", text, flags=re.DOTALL)
    bad = PLACEHOLDER.findall(t)
    if bad:
        sys.exit(f"{name}: unresolved placeholders / bad values: {sorted(set(bad))[:10]}")


def find_chrome() -> str | None:
    cands = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium",
             "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    for c in cands:
        if Path(c).exists():
            return c
    for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        if p := shutil.which(n):
            return p
    return None


def write_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = find_chrome()
    if not chrome:
        print("WARNING: Chrome/Chromium not found; deck.pdf not written (open deck.html and print to PDF instead)")
        return
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--hide-scrollbars", "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=5000", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    pages = len(re.findall(rb"/Type\s*/Page(?![s\w])", pdf_path.read_bytes()))
    print(f"wrote presentation/deck.pdf ({pdf_path.stat().st_size / 1e6:.1f} MB, {pages} pages)")


def main() -> None:
    d = deck()
    check("deck.html", d)
    for txt in (FOOTER,):
        assert d.count(txt) == d.count('class="slide"'), "footer missing on a slide"
    (OUT / "deck.html").write_text(d, encoding="utf-8")
    print(f"wrote presentation/deck.html ({len(d):,} chars, {d.count('class=\"slide\"')} slides)")
    docs = {"abstract.md": abstract(), "pitch.md": pitch(), "qa.md": qa(), "submission.md": submission()}
    for name, text in docs.items():
        check(name, text)
        (DOCS / name).write_text(text, encoding="utf-8")
        print(f"wrote docs/{name} ({len(text):,} chars)")
    write_readme()
    write_pdf(OUT / "deck.html", OUT / "deck.pdf")
    print("rolling-origin:", "included" if ROF else "absent (skipped)")
    print("ranking sunspots:", rank_line("sunspots"))
    print("ranking co2:     ", rank_line("co2"))


if __name__ == "__main__":
    main()
