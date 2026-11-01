# Currency and research method

Edition snapshot: **2026-11-02** (second edition). The first edition's snapshot was 2026-08-21.

This book separates durable mechanisms from volatile product information. Foundational and technical claims point to primary papers in the manuscript bibliography. Security and governance sections point to living official frameworks and legal texts. Named current products appear with an explicit date, mainly in the dated landscape chapter of Part VIII.

## Current pages checked

The dated landscape was checked against the providers' official model catalogs, model cards, changelogs, and documentation on the snapshot date. Each such page is linked inline where it is used, or listed in the bibliography. The starting points were:

- OpenAI model catalog: https://developers.openai.com/api/docs/models
- Anthropic model overview: https://platform.claude.com/docs/en/about-claude/models/overview
- Google Gemini model catalog: https://ai.google.dev/gemini-api/docs/models
- DeepSeek API documentation: https://api-docs.deepseek.com/quick_start/pricing
- Mistral model overview: https://docs.mistral.ai/models
- Hugging Face model hub: https://huggingface.co/models
- Model Context Protocol specification: https://modelcontextprotocol.io/specification
- EUR-Lex, Regulation (EU) 2024/1689 and Regulation (EU) 2026/1744: https://eur-lex.europa.eu/
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- OWASP GenAI Security Project: https://genai.owasp.org/llm-top-10/
- MITRE ATLAS: https://atlas.mitre.org/

These pages are not archived inside the project and may change. Product descriptions are paraphrases of official documentation, not independent performance findings.

## Verification for this edition

Every chapter was written against primary sources and then fact-checked by an independent pass that re-opened the cited source for each claim, recomputed every number, and ran every code example. Every bibliography entry was checked against the arXiv record, Crossref, or the live page.

## Evidence policy

1. Prefer original papers, official documentation, legal texts, and standards bodies.
2. Treat preprints as provisional and link stable identifiers.
3. Do not infer proprietary architecture or training data from product behavior.
4. Avoid point-in-time leaderboard rankings, benchmark scores, and prices.
5. Label rough engineering estimates as estimates and state their assumptions.
6. Distinguish definitions, mathematical consequences, empirical observations, and open interpretations.
7. Remove a claim that cannot be verified on a primary source.
8. For high-stakes deployment, require domain-specific current sources and qualified review beyond this educational book.

## Known limits

- No static book can contain all knowledge or remain permanently current.
- The bibliography is curated rather than exhaustive.
- Many frontier training details are not public.
- Laws, licenses, provider terms, prices, model IDs, and safety policies change.
- The book should be treated as a first-principles map and reading path, not an oracle.
