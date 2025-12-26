template = """Task: Egocentric temporal action localization of "{caption}".
Hint: {seed_start:.4f}-{seed_end:.4f}s.
Output JSON only: {{"scene_summary": "setting", "caption": "{caption}", "start": <start>, "end": <end>, "confidence": <0-1>, "evidence": ["obj"], "notes": "issues"}}
"""
