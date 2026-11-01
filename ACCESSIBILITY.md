# Accessibility

Open LLM is published in three formats so that readers can choose the one that works with their device and assistive technology.

| Format | What it provides |
|---|---|
| Online edition (`/read/`) | Semantic HTML landmarks, a skip link, table header scopes, linked contents and citations, and no scripts |
| EPUB 3 | Reflowable text, semantic headings, structural navigation, a table of contents, descriptive link text, a text alternative for the cover, and display transformability; validated with EPUBCheck |
| A5 PDF | Tagged structure, bookmarks, selectable text, embedded Unicode fonts, and clickable contents and citations |

## Known limits

- Mathematical expressions are written as readable Unicode text, not MathML.
- No formal PDF/UA or WCAG conformance is claimed. Do not read the list above as a conformance statement.

## Report a barrier

If none of the formats works for you, open an issue on the project issue tracker. Please include:

1. the format and the device, operating system, reading application, and assistive technology you used;
2. what you expected and what happened, with the page or section;
3. nothing more personal than you are comfortable sharing.

The maintainers will try to reproduce the problem, suggest another format or reader in the meantime, and fix the source so the next release removes the barrier.
