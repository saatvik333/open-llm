# How Open LLM was written and checked

Open LLM was written by AI language models under the direction of the Open LLM Project. This note says exactly what that means, what was checked, and what was not, so you can decide how much to trust it.

## What was done

1. **Writing.** AI models drafted each part of the book from primary sources: original papers, official documentation, model cards, standards, and legal texts. They were instructed not to state a date, number, name, or attribution without opening a source for it in the same session.
2. **Independent fact-checking.** Separate AI passes, which had not written the text, re-read every chapter. For each cited sentence they opened the cited source and checked that it supports that specific claim: the number, the direction of the effect, the scope, and the date. Errors were fixed in place. Claims that could not be confirmed on a primary source were removed or narrowed.
3. **References.** Each of the 422 bibliography entries was checked against the arXiv record, Crossref, or the live page for existence, authors, title, year, and identifier.
4. **Numbers and code.** Every worked example and derived number was recomputed in Python. Every code block was run, and every printed output in the labs was compared with the actual output.
5. **Consistency.** A final pass read the whole book in order to find contradictions between chapters, broken cross-references, and leftover drafting artifacts.
6. **Publication checks.** The PDF, EPUB, and web edition are built from the same source and validated automatically (`scripts/preflight.py`, `scripts/preflight_release.py`, EPUBCheck).

## What this does not guarantee

- The checking was automated. It reduces errors; it does not prove their absence.
- No human subject-matter expert has reviewed the full text.
- Product names, laws, and capabilities are a dated snapshot (2 November 2026) and will go out of date.
- Preprints can change after publication; the book cites the version current on the snapshot date.

## How to help

If you find a mistake, open an issue with the edition, the section or page, what the book says, and a source for the correction. Corrections are released as new versions; earlier versions stay available under their tags.
