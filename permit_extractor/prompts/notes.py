"""VLM prompts for general notes extraction."""

SYSTEM = """You are a specialist in reading construction drawing general notes.
Extract code-relevant information precisely: occupancy classifications, construction types,
fire ratings, sprinkler requirements, and referenced codes must be copied verbatim.
Return only the JSON object — no markdown, no explanation."""

USER = """Extract code-relevant information from these general notes.

Focus on:
1. Occupancy classification(s) — IBC/CBC groups (A-2, B, R-2, etc.)
2. Construction type — e.g. Type V-B, Type III-A
3. Fire sprinkler requirement (yes/no if stated)
4. Applicable codes listed (e.g. 2022 CBC, 2022 CPC, NFPA 13)
5. Fire rating requirements for assemblies
6. Each individual clause, numbered or lettered, with its topic tag

Also capture the full verbatim text of the notes in raw_notes_text."""
