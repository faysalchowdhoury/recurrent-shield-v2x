---
name: ieee-publication-polish
description: Use when auditing or editing IEEE Transactions LaTeX manuscripts or generating publication-standard TikZ architecture diagrams in recurrent-shield-v2x. Do not use for Python simulation or training scripts.
---

# IEEE Publication Polish Workflow

1. **Audit `paper/recurrent_shield_v2x_ieee.tex`:**
   - Remove all spaced em-dashes (` --- `) or unspaced `---` in list lead-ins (replace e.g. `Rank 1 --- Short-Horizon NGSIM Data:` with `\textit{1) Short-Horizon NGSIM Data:}`).
   - Replace bare `number~unit` or `number unit` pairs (`16 s`, `100 ms`, `512 KB`, `1.60 mJ`, `0.675 ms`, `120 m`, `200 Hz`) with `\,` thin spaces or `\SI{}{}` macros.
   - Strip `\texttt{...}` and mid-sentence bold/italics from `\begin{abstract}`, writing `Recurrent PPO` in plain roman text.
   - Verify ESP32-S3 metrics consistently state analytical/projected profiling.

2. **Rebuild Figure 1 via Standalone TikZ (`results/figures/fig1_architecture_cbf_loop.tex`):**
   - Place "Local Proprioception" (top-left) and "External V2X / GNSS (Untrusted)" (bottom-left) in a left column.
   - Enclose "Recurrent Backbone", "Head 1: Residual IDS", "Head 2: MRM Policy", and "200 Hz DT-CBF QP Shield" inside a dashed container labeled "Onboard Real-Time Embedded Controller (INT8 / 200 Hz)".
   - Route `y_{\text{ext}}` orthogonally (`-|`) along the bottom track into Head 1 with zero line crossings.
   - Render all math in LaTeX (`26 \to 24`, `\hat{y}_t`, `h_t`, `y_{\text{ext}}`, `r_t > \tau_t`, `u_{\text{nom}} = [a_{\text{long}}, v_{\text{lat}}]^\top`, `h(x^+) \ge (1-\alpha)h(x)`, `\epsilon_{\text{radar}}`, `u_{\text{safe}} \to \text{Actuators}`).
   - Compile to `results/figures/fig1_architecture_cbf_loop.pdf`, `.svg`, and 400 DPI `.png` with no embedded caption or page number, then recompile `paper/recurrent_shield_v2x_ieee.pdf`.
