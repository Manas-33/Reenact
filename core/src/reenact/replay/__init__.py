"""Replay: the substitution engine. Matches on (call type, sequence index)
with request-hash verification, supports parallel-window and strict/lenient
modes, and enforces the side-effect policy (mutating tools are always
substituted).
"""
