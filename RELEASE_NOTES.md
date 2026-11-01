# Open LLM — Second Edition, Release 2.0 (v2.0.0)

Release date and technology snapshot: **2 November 2026**. Free and open: text under CC BY 4.0, code under MIT.

## Files

- `Open-LLM-v2.0.0.pdf` — A5, 390 pages, tagged, bookmarked, web-linearized, all fonts embedded
- `Open-LLM-v2.0.0.epub` — reflowable EPUB 3, validated with EPUBCheck 5.2.1 (0 errors, 0 warnings)
- `SHA256SUMS` — verify with `sha256sum -c SHA256SUMS --ignore-missing`
- Web edition: the project site, at `/read/`

## What changed since the first edition (21 August 2026)

- **Brought to the state of the art as of 2 November 2026.** The book grew from about 31,000 to about 98,000 words and from 60 to 422 cited references.
- **Part I:** floating-point and block-scaled number formats (FP8, MX, NVFP4), the roofline model, scale-up and scale-out networks, and matrix-aware optimizers (Muon, Shampoo, SOAP), warmup–stable–decay schedules, and μP.
- **Part II:** training stability (QK-normalization, z-loss, QK-Clip), how embedding models are trained, tokenizer-free and byte-level models, and multi-token and diffusion objectives.
- **Part III:** grouped-query and multi-head latent attention, context extension (position interpolation, YaRN), sliding-window and sparse attention, attention sinks, fine-grained mixture of experts, linear attention and hybrid models, multi-token prediction, diffusion language models, and a dated table of open architectures.
- **Part IV:** model-based data curation, over-training and data-constrained scaling, mid-training, low-precision training, DPO variants, a new chapter on reinforcement learning for reasoning and agents (GRPO and its successors, verifiable rewards, reward hacking, agentic RL), and how long chain-of-thought models are trained and controlled.
- **Part V:** context engineering for long-running agents, agentic retrieval, structured outputs, the Model Context Protocol, Agent2Agent, agent skills, coding, computer-use, and deep-research agents, and current multimodal models, world models, and vision-language-action models.
- **Part VI:** modern serving (chunked prefill, prefix caching, disaggregation, speculative decoding, expert-parallel serving), evaluation of agents and long tasks, prompt-injection defenses by design, frontier safety frameworks, AI control, chain-of-thought monitoring, circuit tracing, and the EU AI Act as amended in 2026.
- **Part VII:** three new labs — budget the key–value cache (Lab 9), reinforce a verifiable answer with GRPO (Lab 10), and run an agent loop under prompt-injection attack (Lab 11). All runnable code was executed and every printed output matches.
- **Part VIII:** a new dated landscape of hosted and open-weight model families as of 2 November 2026, and refreshed research directions.
- **Appendices:** formula sheet, glossary (about 270 terms), myths, checklists, and curriculum updated to match.
- **Corrections to the first edition:** about 30 factual, citation, and code fixes found in an audit before this release.
- **Distribution:** the book is now free and open, published as GitHub Release assets and a static reading site; the earlier private paid-launch tooling was removed.

## How this edition was made and checked

The text was written by AI language models under the direction of the Open LLM Project and checked by separate AI fact-checking passes against the cited primary sources; see `VERIFICATION.md` for the method and its limits. No human subject-matter expert has reviewed the full text. Please report errors with the issue form.

## Verification record

`./scripts/build-digital-release.sh` passed on the release build:

- manuscript and PDF preflight: 10 manuscript files, continuous references R01–R422 all cited, headings and PDF bookmarks matching exactly, contents page numbers matching PDF destinations, every Python example parsed and every JSON example loaded, qpdf structure check, 11 embedded fonts;
- EPUBCheck 5.2.1 under EPUB 3.3 rules: 0 errors, 0 warnings;
- release preflight: versioned file names, SHA-256 checksums, `CITATION.cff` and `book.json` agreement, static script-free site with restrictive security headers;
- browser audit of the site: no remote requests, CSP violations, broken links, or heading-level jumps.

## Toolchain

Node.js 26.12, Playwright 1.64.1 with Chromium, qpdf 12.4.2, Poppler, Python 3.14, EPUBCheck 5.2.1, ImageMagick 7.

## Known limits

Product names, laws, and capabilities are a snapshot dated 2 November 2026. No ISBN or formal PDF/UA conformance is claimed. A print-on-demand service may need a printer-specific interior and cover.
