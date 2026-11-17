"""Uninstructed collusion: all four agents get the same system prompt, and whether a pair colludes is decided inside the
episode. Design and pre-registration: docs/uninstructed_collusion/DESIGN.md.

  protocol.py   the symmetric episode (two private rooms, three public rounds, vote) and the stake versions S0 to S3
  stakes.json   the stake texts, one pair of sentences per scenario
  coding.py     the rubric for coding a room, the label rule and agreement statistics
  timeline.py   replay of room turns and per-token reads
"""
