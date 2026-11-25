# RecurrentShield-V2X Workspace Rules (IEEE T-ITS Standards)

When editing or verifying files inside this repository, strictly follow these rules:

1. **Method & Provenance Honesty:** Never describe supervised multi-task behavior cloning (`RecurrentShieldBC` / `StatelessBC`) as PPO; never claim physical radar/IMU or raw RF SPMD logs when evaluating trajectory-grounded USDOT NGSIM data mapped to SAE J2735 BSM fields; always distinguish 10\,Hz outer policy updates from 200\,Hz substepped nonlinear DT-CBF checks and analytical ESP32-S3 estimates.
2. **Prose & Punctuation Hygiene:** Never place spaced em-dashes (` --- `) or unspaced `---` inside English prose or list-item headings. Convert `Rank 1 --- ...:` / `Rank 1 — ...:` headers cleanly to `\textit{Title:}` inside `\begin{enumerate}`. Keep `--` strictly for numeric ranges (`0.1--1.5`).
3. **SI & Compute Unit Spacing:** Use non-breaking thin spaces (`\,`) between numbers and units across prose, equations, tables, and captions (e.g., `10\,Hz`, `200\,Hz`, `82.6\,KB`, `107.1\,KB`, `1.70\,ms`, `16.0\,\mu\text{s}`).
4. **Abstract Cleanliness:** Keep `\begin{abstract}` plain roman text—zero `\texttt{...}`, `\textbf{...}`, or `\emph{...}` macros.
5. **Architecture Diagram Standards:** Compile [results/figures/fig1_architecture_cbf_loop.tex](file:///C:/Users/sheel/Downloads/recurrent-shield-v2x/results/figures/fig1_architecture_cbf_loop.tex) via standalone TikZ (`newtxtext`, `newtxmath`), pure white `#FFFFFF` backdrop, zero line crossings, and proper math-mode symbols.
