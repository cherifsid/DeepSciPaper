# Research Copilot Skill Pack

This folder contains human-authored playbooks that the Deep Search agent (GPT Researcher) loads at runtime and uses as its operating procedure when:

- generating search queries
- choosing sources and retrievers
- extracting evidence
- writing the final research report

The app concatenates these markdown files into a single "agent role" instruction prompt and passes it to GPT Researcher as `role=...`.

Design goals:
- reproducible, evidence-grounded reports
- bias-aware coverage (not only arXiv, not only web)
- high-precision citations and page-level traceability for PDFs when possible
- stable behavior across different retrievers (DDG, arXiv, etc.)

