"""Counterpoint — corpus hypothesis validation (PoC, step 1).

Step 1 scope: ingest + semantic chunking + atomization + cached lift grid
+ validation matrix. Conjunction (MUS) extraction and residual detection are
later steps and deliberately not implemented here.

Terminology is fixed throughout:
  data point  — an atomic, single-proposition unit of the corpus
  hypothesis  — a proposition validated against the corpus
No other sense of either word is used.
"""
__version__ = "0.1.0"
