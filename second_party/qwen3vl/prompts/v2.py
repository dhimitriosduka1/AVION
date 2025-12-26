template = """TASK: Temporal localization in egocentric video.

ACTION TO LOCATE: "{caption}"

SEED WINDOW (approximate): {seed_start:.4f}s to {seed_end:.4f}s (use as starting point only, may be inaccurate).

ANALYSIS STEPS:
1. Watch the video and identify the camera wearer's hands throughout.
2. Find when the described action STARTS: the exact moment of first intentional movement toward the action (hand reaches, begins grasp, or object starts moving due to wearer).
3. Find when the action ENDS: the exact moment the action goal is achieved (object released/placed, hands withdraw, result is stable).

VISUAL CUES TO TRACK:
- Hand position and motion relative to target objects
- Object state changes (picked up, moved, opened, closed, placed)
- Contact events (hand touches object, object touches surface)
- Motion ends (object comes to rest, hand stops moving)

CRITICAL RULES:
- Boundaries must be TIGHT: start at first evidence of action, end when action completes
- Do NOT include preparation or aftermath unless part of the described action
- If action spans multiple sub-actions, include the full sequence
- Times are relative to video start (0.0s = first frame)

OUTPUT (JSON only, no explanation):
{{"scene_summary": "<20 words describing setting>", "caption": "{caption}", "start": <seconds>, "end": <seconds>, "confidence": <0.0-1.0>, "evidence": ["<object1>", "<object2>"], "notes": "<brief uncertainty if any>"}}
END"""
