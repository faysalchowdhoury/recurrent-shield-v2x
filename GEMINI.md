# RecurrentShield-V2X — IEEE T-ITS Standards

## Manuscript & Prose Constraints
- Never use spaced em-dashes (` --- `). Only `IEEEtran` built-in `Abstract---` and `Index Terms---` use unspaced em-dashes; use colons or periods for list headers.
- Always use `siunitx` (`\SI{200}{\hertz}`, `\SI{82.6}{\kilo\byte}`) or non-breaking thin spaces (`16\,s`, `100\,ms`) between numbers and units so justified columns never stretch spaces before units.
- Never put `\texttt{...}`, bold-italics, or raw ASCII code strings (`->`, `>=`, `y_hat`) inside `\begin{abstract}` or figure labels.
- Keep ESP32-S3 cycle, latency, and energy figures explicitly labeled as analytical/projected per Rank 3.

## Architecture Figure Constraints
- Never generate system architecture block diagrams with Matplotlib.
- Build architecture figures with standalone LaTeX TikZ (`\documentclass[border=6pt]{standalone}`) using `newtxtext` and `newtxmath` on a pure `#FFFFFF` background.
- Route external untrusted signals (`y_{\text{ext}}`) orthogonally (`-|`) along clean tracks without crossing node boxes or colliding with other arrows.
- Export to vector `.pdf` / `.svg` and 400 DPI `.png` without embedding figure captions or page numbers inside the diagram image.
