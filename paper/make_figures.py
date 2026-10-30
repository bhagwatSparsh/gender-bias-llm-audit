"""Make all paper figures and LaTeX table fragments from the CSV files in data/.
Run:  python3 make_figures.py      (writes figures/*.pdf and tables/*.tex)"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import binomtest

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 7})
TEAL, CORAL, INK, MUTE, GRID = "#0090A8", "#E4572E", "#1F2933", "#6B7280", "#E5E7EB"
MODELS = ["GPT-OSS-120B", "GPT-OSS-20B", "Qwen-3.8-27B", "Gemini-3.5-Flash-Lite"]
SHORT = {"GPT-OSS-120B": "GPT-OSS-120B", "GPT-OSS-20B": "GPT-OSS-20B", "Qwen-3.8-27B": "Qwen-3.8-27B",
         "Gemini-3.5-Flash-Lite": "Gemini-3.5-FL"}
P = pd.read_csv("data/primary.csv")
PR = P[P.family == "primary"]
ALPHA = 0.05

def col(v):
    return TEAL if v > 0 else (CORAL if v < 0 else MUTE)

def dot(ax, x, lo, hi, y, shape, color, filled, lw=1.3, ms=4.6):
    ax.plot([lo, hi], [y, y], color=color, lw=lw + (0.5 if filled else 0), solid_capstyle="butt", zorder=2)
    ax.plot(x, y, shape, ms=ms, mfc=color if filled else "white", mec=color, mew=1.1, zorder=3)

# ---------------------------------------------------------------- forest plots (approval + score)
def forest_panel(ax, outcome, ctx, xlim, show_ylabels, unit_note=None, annotate=True):
    d = PR[(PR.outcome == outcome) & (PR.context == ctx)]
    yt, yl = [], []
    for i, m in enumerate(MODELS):
        for j, (fr, shp) in enumerate([("implicit", "o"), ("explicit", "s")]):
            r = d[(d.model_name == m) & (d.framing == fr)].iloc[0]
            y = i * 2.3 + j * 0.8
            sig = r.p_holm < ALPHA
            dot(ax, r["mean"], r.ci_lo, r.ci_hi, y, shp, col(r["mean"]), sig)
            if sig and annotate:
                ax.text(min(r.ci_hi, xlim[1]) + (xlim[1] - xlim[0]) * 0.02, y,
                        f"Holm $p$ = {r.p_holm:.3f}", va="center", ha="left", fontsize=6.8, color=INK)
        yt.append(i * 2.3 + 0.4); yl.append(SHORT[m])
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(*xlim); ax.set_ylim(len(MODELS) * 2.3 - 0.9, -0.9)
    ax.set_yticks(yt); ax.set_yticklabels(yl if show_ylabels else [])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)

def make_forest():
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 3.9))
    specs = [("approval_gap_pp(M-F)", (-30, 24), "Approval gap, men minus women (percentage points)"),
             ("suitability_gap_points(M-F)", (-0.9, 0.75), "Score gap, men minus women (points on 1–10 scale)")]
    for r, (out, xl, lab) in enumerate(specs):
        for c, ctx in enumerate(["general", "india"]):
            ax = axs[r, c]
            forest_panel(ax, out, ctx, xl, show_ylabels=(c == 0))
            ax.set_xlabel(lab if True else "")
            tag = "abcd"[r * 2 + c]
            ax.set_title(f"({tag}) {'Approval' if r == 0 else 'Score'} – {'general' if ctx == 'general' else 'India'} context",
                         loc="left", fontsize=7.8)
    h = [Line2D([], [], marker="o", color=MUTE, lw=1.3, mfc="white", mec=MUTE, ms=4.6, label="name only"),
         Line2D([], [], marker="s", color=MUTE, lw=1.3, mfc="white", mec=MUTE, ms=4.6, label="name + pronouns"),
         Line2D([], [], marker="o", color=MUTE, lw=1.8, mfc=MUTE, mec=MUTE, ms=4.6, label="filled: Holm $p$ < 0.05"),
         Line2D([], [], marker="o", color=TEAL, lw=1.3, mfc="white", mec=TEAL, ms=4.6, label="teal: men favoured"),
         Line2D([], [], marker="o", color=CORAL, lw=1.3, mfc="white", mec=CORAL, ms=4.6, label="coral: women favoured")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=1.1, w_pad=1.0)
    fig.savefig("figures/fig_forest.pdf"); plt.close(fig)

# ---------------------------------------------------------------- India vs general and naming effect
def make_context_naming():
    C = pd.read_csv("data/context_comparison.csv"); A = pd.read_csv("data/anon_baseline.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 3.05), gridspec_kw={"width_ratios": [1, 1.35]})
    d = C[C.outcome.str.startswith("approval")]
    for i, m in enumerate(MODELS):
        r = d[d.model_name == m].iloc[0]
        dot(a1, r["mean"], r.ci_lo, r.ci_hi, i, "o", col(r["mean"]), r.p_holm < ALPHA, ms=5.2)
        if r.p_holm < ALPHA:
            a1.text(r["mean"], i + 0.32, f"Holm $p$ = {r.p_holm:.3f}", va="center", ha="center", fontsize=6.8)
    a1.axvline(0, color=INK, lw=0.7, ls=(0, (3, 2)))
    a1.set_yticks(range(4)); a1.set_yticklabels([SHORT[m] for m in MODELS]); a1.invert_yaxis()
    a1.set_xlim(-28, 24); a1.tick_params(axis="y", length=0); a1.grid(axis="x", color=GRID, lw=0.5); a1.set_axisbelow(True)
    a1.set_xlabel("Approval gap in India minus gap in general context (pp)")
    a1.set_title("(a) Does the gap depend on context?", loc="left", fontsize=7.8)
    ap = A[A.outcome.str.startswith("approval")]
    yt, yl, y = [], [], 0
    for m in MODELS:
        for ctx in ["general", "india"]:
            for k, (g, shp, colr) in enumerate([("male", "o", TEAL), ("female", "s", CORAL)]):
                r = ap[(ap.model_name == m) & (ap.context == ctx) & (ap.outcome == f"approval: {g} - anonymous")].iloc[0]
                dot(a2, r["mean"], r.ci_lo, r.ci_hi, y + k * 0.42, shp, colr, r.p_holm < ALPHA)
            yt.append(y + 0.21); yl.append(f"{SHORT[m]} · {'general' if ctx == 'general' else 'India'}")
            y += 1.2
        y += 0.35
    a2.axvline(0, color=INK, lw=0.7, ls=(0, (3, 2)))
    a2.set_yticks(yt); a2.set_yticklabels(yl, fontsize=6.6); a2.invert_yaxis(); a2.tick_params(axis="y", length=0)
    a2.set_xlim(-32, 18); a2.grid(axis="x", color=GRID, lw=0.5); a2.set_axisbelow(True)
    a2.set_xlabel("Approval with a name minus approval with no name (pp)")
    a2.set_title("(b) Name effect versus the anonymous baseline", loc="left", fontsize=7.8)
    h = [Line2D([], [], marker="o", color=TEAL, lw=1.3, mfc="white", mec=TEAL, ms=4.6, label="male name"),
         Line2D([], [], marker="s", color=CORAL, lw=1.3, mfc="white", mec=CORAL, ms=4.6, label="female name"),
         Line2D([], [], marker="o", color=MUTE, lw=1.8, mfc=MUTE, mec=MUTE, ms=4.6, label="filled: Holm $p$ < 0.05")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.6); fig.savefig("figures/fig_context_naming.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Study 3 stereotype tests
def make_stereo():
    R = pd.read_csv("data/stereo_results.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=True)
    specs = [("H1", 50, 100, "Resolved pronouns that follow\nthe stereotype (%)", "(a) Pronoun test", (35, 92), True),
             ("H3", 0, 100, "P(male-stereotyped role | male name)\nminus P(… | female name) (pp)", "(b) Role recommendation", (-9, 18), False),
             ("H2", 0, 1, "Agentic minus communal words per 100,\nmale minus female letters", "(c) Reference letters", (-0.45, 0.6), False)]
    for ax, (h, base, scale, lab, title, xl, neutral) in zip(axs, specs):
        d = R[R.hypothesis == h].set_index("model_name")
        for i, m in enumerate(MODELS):
            r = d.loc[m]; mean, lo, hi = base + scale * r["mean"], base + scale * r.ci_lo, base + scale * r.ci_hi
            c = INK if neutral else col(r["mean"])
            dot(ax, mean, lo, hi, i, "o", c, r.p_holm < ALPHA, ms=5.0)
            tag = f"p$_{{Holm}}$ = {r.p_holm:.3f}" if r.p_holm < 0.1 else ""
            if tag:
                ax.text(min(hi + (xl[1] - xl[0]) * 0.02, xl[1] - (xl[1] - xl[0]) * 0.3), i - 0.33, tag, fontsize=6.3, va="center")
        ax.axvline(base, color=INK, lw=0.7, ls=(0, (3, 2)))
        ax.set_xlim(*xl); ax.set_ylim(3.6, -0.6); ax.set_yticks(range(4)); ax.set_yticklabels([SHORT[m] for m in MODELS])
        ax.tick_params(axis="y", length=0); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_xlabel(lab, fontsize=7); ax.set_title(title, loc="left", fontsize=7.8)
    h = [Line2D([], [], marker="o", color=MUTE, lw=1.8, mfc=MUTE, mec=MUTE, ms=4.6, label="filled: Holm $p$ < 0.05"),
         Line2D([], [], marker="o", color=MUTE, lw=1.3, mfc="white", mec=MUTE, ms=4.6, label="hollow: not reliable"),
         Line2D([], [], marker="o", color=TEAL, lw=1.3, mfc="white", mec=TEAL, ms=4.6, label="teal: male direction"),
         Line2D([], [], marker="o", color=CORAL, lw=1.3, mfc="white", mec=CORAL, ms=4.6, label="coral: female direction")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8); fig.savefig("figures/fig_stereo.pdf"); plt.close(fig)

# ---------------------------------------------------------------- case-level male-bias map
def counts():
    d = pd.read_excel("data/discordant_main.xlsx", sheet_name="Decision differs")
    d = d[d.framing != "placebo"]
    r = pd.read_excel("data/discordant_stereo.xlsx", sheet_name=0)
    rows = {}
    def cnt(df, mcol, key_m, key_f):
        return int((df[mcol] == key_m).sum()), int((df[mcol] == key_f).sum())
    out = {}
    for m in MODELS:
        dm = d[d.model_name == m]; rm = r[r.model == m]
        for dom in ["hiring", "promotion", "firing"]:
            out[(m, dom.capitalize())] = cnt(dm[dm.domain == dom], "favoured_by_decision", "male", "female")
        for ctx in ["general", "india"]:
            out[(m, "D" + ctx)] = cnt(dm[dm.context == ctx], "favoured_by_decision", "male", "female")
            out[(m, "R" + ctx)] = cnt(rm[rm.context == ctx], "bias_direction", "male-leaning", "female-leaning")
        out[(m, "Dall")] = cnt(dm, "favoured_by_decision", "male", "female")
        out[(m, "Rall")] = cnt(rm, "bias_direction", "male-leaning", "female-leaning")
    return out

def make_map():
    out = counts()
    rows = [("Hiring", "Hiring"), ("Promotion", "Promotion"), ("Firing", "Firing (retain = favourable)"),
            ("Ddgeneral", None)]
    layout = [("Decisions that differed", None),
              ("Hiring", "Hiring"), ("Promotion", "Promotion"), ("Firing", "Firing"),
              ("Dgeneral", "General context"), ("Dindia", "India context"), ("Dall", "All decisions"),
              ("Role picks that differed", None),
              ("Rgeneral", "General context"), ("Rindia", "India context"), ("Rall", "All role picks")]
    short = ["GPT-OSS\n120B", "GPT-OSS\n20B", "Qwen\n3.8-27B", "Gemini\n3.5-FL"]
    cmap = LinearSegmentedColormap.from_list("div", [CORAL, "#F5F5F4", TEAL])
    fig, ax = plt.subplots(figsize=(3.5, 3.55))
    ax.set_xlim(-1.55, 4); ax.set_ylim(len(layout) - 0.4, -1.15); ax.axis("off")
    pvals = {}
    for r_i, (key, lab) in enumerate(layout):
        if lab is None:
            ax.text(-1.5, r_i, key, fontsize=7.3, fontweight="bold", va="center"); continue
        ax.text(-1.5, r_i, lab, fontsize=7.2, va="center", fontweight="bold" if key in ("Dall", "Rall") else "normal")
        for c_i, m in enumerate(MODELS):
            a, b = out[(m, key)]
            diff = a - b
            ax.add_patch(plt.Rectangle((c_i - 0.47, r_i - 0.42), 0.94, 0.84, fc=cmap(0.5 + np.clip(diff, -20, 20) / 40.0),
                                       ec="white", lw=1.2))
            p = binomtest(a, a + b, 0.5).pvalue if a + b > 0 else 1.0
            pvals[(m, key)] = p
            star = "*" if (a + b >= 6 and p < 0.05) else ""
            ax.text(c_i, r_i, f"{a} / {b}{star}", ha="center", va="center", fontsize=7.0, color=INK,
                    fontweight="bold" if key in ("Dall", "Rall") else "normal")
    for c_i, s in enumerate(short):
        ax.text(c_i, -0.85, s, ha="center", va="center", fontsize=6.8, linespacing=0.95)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig("figures/fig_map.pdf"); plt.close(fig)
    return out, pvals

# ---------------------------------------------------------------- LaTeX table fragments
def n(x, d=1):
    s = f"{x:.{d}f}"
    return f"${s}$" if x >= 0 else "$-$" + f"${s[1:]}$"
def num(x, d=1):
    s = f"{abs(x):.{d}f}"
    return ("$-" if x < -0.5 * 10 ** (-d) else "$") + s + "$"
def ci(lo, hi, d=1):
    return f"[{num(lo, d)}, {num(hi, d)}]"
def pf(p):
    if p != p: return "--"
    if p < 0.001: return "$<$0.001"
    if p >= 0.995: return "1.00"
    return f"{p:.3f}"

def tab_primary():
    names = {"approval_gap_pp(M-F)": ("A1", "approval gap (percentage points)", 1),
             "suitability_gap_points(M-F)": ("A2", "score gap (points)", 2),
             "value_gap_pct(M-F)": ("A3", "value gap (\\% of amount)", 1)}
    for o, (tag, cap, d) in names.items():
        t = PR[PR.outcome == o]
        L = ["\\begin{tabular}{@{}llcrrrrr@{}}", "\\toprule",
             "Model & Context & Framing & $n$ & Gap & 95\\% CI & mde80 & Holm $p$ \\\\", "\\midrule"]
        for m in MODELS:
            for ctx in ["general", "india"]:
                for fr in ["implicit", "explicit"]:
                    r = t[(t.model_name == m) & (t.context == ctx) & (t.framing == fr)].iloc[0]
                    cell = num(r["mean"], d)
                    if r.p_holm < ALPHA: cell = "\\textbf{" + cell + "}"
                    L.append(f"{SHORT[m]} & {'general' if ctx=='general' else 'India'} & {'name only' if fr=='implicit' else 'name + pron.'} & {int(r.n)} & {cell} & {ci(r.ci_lo, r.ci_hi, d)} & {r.mde80:.{d}f} & {pf(r.p_holm)} \\\\")
            if m != MODELS[-1]: L.append("\\addlinespace[1pt]")
        L += ["\\bottomrule", "\\end{tabular}"]
        open(f"tables/tab_primary_{tag}.tex", "w").write("\n".join(L) + "\n")

def tab_pooled():
    pooled = P[P.framing == "pooled"]
    L = ["\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
         "Model & Context & Approval gap (pp) & Score gap (points) & Value gap (\\%) \\\\", "\\midrule"]
    for m in MODELS:
        for ctx in ["general", "india"]:
            cells = []
            for o, d in [("approval_gap_pp(M-F)", 1), ("suitability_gap_points(M-F)", 2), ("value_gap_pct(M-F)", 1)]:
                r = pooled[(pooled.outcome == o) & (pooled.model_name == m) & (pooled.context == ctx)].iloc[0]
                s = f"{num(r['mean'], d)} {ci(r.ci_lo, r.ci_hi, d)}"
                cells.append(s)
            L.append(f"{SHORT[m]} & {'general' if ctx=='general' else 'India'} & " + " & ".join(cells) + " \\\\")
        if m != MODELS[-1]: L.append("\\addlinespace[1pt]")
    L += ["\\bottomrule", "\\end{tabular}"]
    open("tables/tab_pooled.tex", "w").write("\n".join(L) + "\n")

def tab_validity():
    V = pd.read_csv("data/validity.csv"); V = V[V.context == "all"].set_index("model_name")
    L = ["\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         "Model & Clear yes & Clear no & Borderline & Informative \\\\", "\\midrule"]
    for m in MODELS:
        r = V.loc[m]
        L.append(f"{SHORT[m]} & {100*r.strong:.0f} & {100*r.weak:.0f} & {100*r.borderline:.1f} & {100*r.borderline_informative_frac:.1f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open("tables/tab_validity.tex", "w").write("\n".join(L) + "\n")

def tab_stereo():
    R = pd.read_csv("data/stereo_results.csv")
    prim = R[R["primary"] == True]
    lab = {"H1": ("H1 pronoun: share consistent $-$ 0.5", 3), "H3": ("H3 role: male $-$ female name", 3),
           "H2": ("H2 letters: agentic $-$ communal", 3)}
    L = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
         "Test & Model & $n$ & Estimate & 95\\% CI & mde80 & Holm $p$ \\\\", "\\midrule"]
    for h in ["H1", "H3", "H2"]:
        for m in MODELS:
            r = prim[(prim.hypothesis == h) & (prim.model_name == m)].iloc[0]
            cell = num(r["mean"], 3)
            if r.p_holm < ALPHA: cell = "\\textbf{" + cell + "}"
            L.append(f"{lab[h][0] if m == MODELS[0] else ''} & {SHORT[m]} & {int(r.n)} & {cell} & {ci(r.ci_lo, r.ci_hi, 3)} & {r.mde80:.3f} & {pf(r.p_holm)} \\\\")
        if h != "H2": L.append("\\addlinespace[2pt]")
    L += ["\\bottomrule", "\\end{tabular}"]
    open("tables/tab_stereo.tex", "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    make_forest(); make_context_naming(); make_stereo()
    out, pv = make_map()
    tab_primary(); tab_pooled(); tab_validity(); tab_stereo()
    print("cells with uncorrected sign-test p<0.05 (>=6 discordant):")
    for (m, k), p in pv.items():
        a, b = out[(m, k)]
        if a + b >= 6 and p < 0.05: print(" ", m, k, a, b, round(p, 4))
    tot = {k: (sum(out[(m, k)][0] for m in MODELS), sum(out[(m, k)][1] for m in MODELS)) for k in ["Dall", "Rall"]}
    for k, (a, b) in tot.items(): print(k, a, b, round(binomtest(a, a + b, .5).pvalue, 4))
    for m in MODELS: print(m, out[(m, "Dall")], round(pv[(m, "Dall")], 3), out[(m, "Rall")], round(pv[(m, "Rall")], 3))
