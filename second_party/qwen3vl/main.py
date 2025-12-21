import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from statistics import median, mean

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


MODEL_PATH = "Qwen/Qwen3-VL-8B-Thinking"


# =========================
# Data structures
# =========================


@dataclass
class WindowSpec:
    start: float  # absolute seconds from video start
    end: float  # absolute seconds from video start
    half_span: float  # (end - start) / 2


@dataclass
class LocalizationResult:
    present: bool
    start: float  # seconds (here: absolute from video start)
    end: float  # seconds (absolute)
    confidence: Optional[float]
    rationale: Optional[str]
    raw_text: str
    window: WindowSpec
    is_refined: bool = False
    reasoning_chain: Optional[str] = None
    hypotheses_count: Optional[int] = None


# =========================
# Model / processor loading
# =========================


def _load_qwen3vl(query_text: str = ""):
    """
    Load Qwen3-VL model + processor with adaptive resolution.
    Higher resolution for fine-grained actions.
    """
    print(f"Loading model: {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    # Adaptive resolution based on query complexity
    fine_grained_keywords = [
        "check",
        "examine",
        "inspect",
        "look",
        "adjust",
        "arrange",
        "pick",
        "place",
        "touch",
        "grab",
    ]

    is_fine_grained = any(word in query_text.lower() for word in fine_grained_keywords)

    if is_fine_grained:
        min_pixels = 16 * 32 * 32  # Higher resolution for detailed actions
        max_pixels = 512 * 32 * 32
        print("Using high resolution for fine-grained action detection")
    else:
        min_pixels = 4 * 32 * 32
        max_pixels = 256 * 32 * 32
        print("Using standard resolution")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return model, processor


# =========================
# Parsing helpers
# =========================


def _safe_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Grab the first {...} block and parse it as JSON."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    return _safe_json(m.group(0))


def _extract_reasoning(text: str) -> Optional[str]:
    """Extract reasoning/analysis before JSON block."""
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        reasoning = text[: json_match.start()].strip()
        return reasoning if reasoning else None
    return None


def _parse_timecodes_or_numbers(text: str):
    """
    Fallback parser:
      - hh:mm:ss(.ms) or mm:ss(.ms)
      - or first two bare numbers
    Returns (start_sec, end_sec) or None.
    """
    # hh:mm:ss(.ms) or mm:ss(.ms)
    timecodes = re.findall(r"\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2}(?:\.\d+)?)\b", text)

    def to_seconds(h, m, s):
        h = int(h) if h else 0
        m = int(m)
        sec = float(s)
        return h * 3600 + m * 60 + sec

    if len(timecodes) >= 2:
        s = to_seconds(*timecodes[0])
        e = to_seconds(*timecodes[1])
        return s, e

    # Bare numbers
    nums = re.findall(r"(\d+(?:\.\d+)?)", text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])

    return None


def parse_localization_output(
    text: str,
    window: WindowSpec,
    assume_present_if_missing: bool = True,
) -> Optional[LocalizationResult]:
    """
    Parse Qwen's output into a LocalizationResult.
    Extracts both JSON data and reasoning chain.
    """
    obj = _extract_json_block(text)
    reasoning = _extract_reasoning(text)

    present = True
    start = None
    end = None
    confidence = None
    rationale = None

    if obj is not None:
        if "present" in obj:
            present = bool(obj["present"])
        elif not assume_present_if_missing:
            present = False

        if "start" in obj and "end" in obj:
            try:
                start = float(obj["start"])
                end = float(obj["end"])
            except Exception:
                start = end = None

        if "confidence" in obj:
            try:
                confidence = float(obj["confidence"])
            except Exception:
                confidence = None

        if "rationale" in obj:
            rationale = str(obj["rationale"])

    # Fallback if needed
    if start is None or end is None:
        pair = _parse_timecodes_or_numbers(text)
        if pair is None:
            return None
        start, end = pair
        present = True

    if end < start:
        return None
    if end - start <= 0:
        return None

    return LocalizationResult(
        present=present,
        start=start,
        end=end,
        confidence=confidence,
        rationale=rationale,
        raw_text=text,
        window=window,
        reasoning_chain=reasoning,
    )


# =========================
# Single-pass full-video call
# =========================


def _run_qwen_full_video(
    model,
    processor,
    video_path: str,
    prompt_text: str,
    fps: float,
    max_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """
    Run Qwen3-VL on the full video with configurable generation parameters.
    """
    video_item = {
        "type": "video",
        "video": video_path,
        "fps": float(fps),
    }

    messages = [
        {
            "role": "user",
            "content": [
                video_item,
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Text prompt
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Vision preprocessing with metadata for time alignment
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos = list(videos)
        video_metadatas = list(video_metadatas)
    else:
        videos = None
        video_metadatas = None

    proc_kwargs = dict(
        text=[text],
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        do_resize=False,
    )
    if video_kwargs:
        proc_kwargs.update(video_kwargs)

    inputs = processor(**proc_kwargs).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Strip prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return out_text


# =========================
# Enhanced localization strategies
# =========================


def get_multiple_hypotheses(
    model,
    processor,
    video_path: str,
    query_text: str,
    egovlp_start: float,
    egovlp_end: float,
    fps: float,
    window: WindowSpec,
    n_samples: int = 3,
) -> List[LocalizationResult]:
    """
    Generate multiple boundary predictions using sampling for diversity.
    Aggregate to get more robust estimates.
    """
    print(f"[Multi-hypothesis] Generating {n_samples} diverse predictions...")

    prompt = (
        f'Event: "{query_text}"\n'
        f"EgoVLP prediction: {egovlp_start:.2f}-{egovlp_end:.2f} seconds.\n\n"
        "Analyze the video and determine when this event occurs.\n"
        "Output ONLY JSON:\n"
        '{"present": true/false, "start": <float>, "end": <float>, '
        '"confidence": <0-1>, "rationale": "..."}\n'
        "Times are in seconds from video start."
    )

    hypotheses = []
    for i in range(n_samples):
        raw_output = _run_qwen_full_video(
            model=model,
            processor=processor,
            video_path=video_path,
            prompt_text=prompt,
            fps=fps,
            max_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        result = parse_localization_output(raw_output, window)
        if (
            result
            and result.present
            and result.start >= 0
            and result.end > result.start
        ):
            hypotheses.append(result)
            print(
                f"  Hypothesis {i+1}: [{result.start:.2f}, {result.end:.2f}], "
                f"conf={result.confidence}"
            )

    return hypotheses


def aggregate_hypotheses(
    hypotheses: List[LocalizationResult], method: str = "weighted_median"
) -> Optional[LocalizationResult]:
    """
    Aggregate multiple hypotheses into a single result.
    Methods: 'median', 'mean', 'weighted_median', 'highest_confidence'
    """
    if not hypotheses:
        return None

    if len(hypotheses) == 1:
        return hypotheses[0]

    if method == "highest_confidence":
        return max(hypotheses, key=lambda h: h.confidence or 0.0)

    elif method == "median":
        starts = [h.start for h in hypotheses]
        ends = [h.end for h in hypotheses]
        agg_start = median(starts)
        agg_end = median(ends)

    elif method == "mean":
        starts = [h.start for h in hypotheses]
        ends = [h.end for h in hypotheses]
        agg_start = mean(starts)
        agg_end = mean(ends)

    elif method == "weighted_median":
        # Weight by confidence
        weighted_starts = []
        weighted_ends = []

        for h in hypotheses:
            weight = h.confidence if h.confidence else 0.5
            weighted_starts.extend([h.start] * int(weight * 10))
            weighted_ends.extend([h.end] * int(weight * 10))

        agg_start = (
            median(weighted_starts)
            if weighted_starts
            else median([h.start for h in hypotheses])
        )
        agg_end = (
            median(weighted_ends)
            if weighted_ends
            else median([h.end for h in hypotheses])
        )

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    # Use highest confidence from all hypotheses
    best_conf = max(h.confidence for h in hypotheses if h.confidence is not None)

    # Combine rationales
    combined_rationale = (
        f"Aggregated from {len(hypotheses)} hypotheses. "
        + "; ".join([h.rationale for h in hypotheses if h.rationale])[:200]
    )

    return LocalizationResult(
        present=True,
        start=agg_start,
        end=agg_end,
        confidence=best_conf,
        rationale=combined_rationale,
        raw_text=f"Aggregated from {len(hypotheses)} predictions",
        window=hypotheses[0].window,
        hypotheses_count=len(hypotheses),
    )


def grounded_localization_with_tracking(
    model,
    processor,
    video_path: str,
    query_text: str,
    fps: float,
    window: WindowSpec,
) -> Optional[LocalizationResult]:
    """
    Use object/action tracking for more precise localization.
    """
    print("[Grounded tracking] Analyzing with object/action focus...")

    prompt = (
        f'Event description: "{query_text}"\n\n'
        "Analyze this video step-by-step:\n"
        "1. Identify the KEY OBJECT or ACTION mentioned in the event\n"
        "2. Track when this object/action FIRST APPEARS in the video\n"
        "3. Track when this object/action LAST APPEARS or the action COMPLETES\n"
        "4. Determine precise timestamps\n\n"
        "Think through each step, then output JSON:\n"
        '{"key_element": "description of main object/action",\n'
        ' "first_seen": <float seconds from video start>,\n'
        ' "last_seen": <float seconds from video start>,\n'
        ' "confidence": <0-1>,\n'
        ' "tracking_notes": "what you observed"}\n'
    )

    raw_output = _run_qwen_full_video(
        model=model,
        processor=processor,
        video_path=video_path,
        prompt_text=prompt,
        fps=fps,
        max_tokens=384,
        do_sample=False,
    )

    obj = _extract_json_block(raw_output)
    reasoning = _extract_reasoning(raw_output)

    if obj and "first_seen" in obj and "last_seen" in obj:
        try:
            start = float(obj["first_seen"])
            end = float(obj["last_seen"])
            confidence = float(obj.get("confidence", 0.7))

            rationale = (
                f"Tracked: {obj.get('key_element', 'object/action')}. "
                + obj.get("tracking_notes", "")
            )

            print(f"  Tracked element: {obj.get('key_element', 'unknown')}")
            print(f"  First seen: {start:.2f}s, Last seen: {end:.2f}s")

            return LocalizationResult(
                present=True,
                start=start,
                end=end,
                confidence=confidence,
                rationale=rationale,
                raw_text=raw_output,
                window=window,
                reasoning_chain=reasoning,
            )
        except (ValueError, KeyError) as e:
            print(f"  Failed to parse tracking output: {e}")
            return None

    return None


def coarse_with_reasoning(
    model,
    processor,
    video_path: str,
    query_text: str,
    egovlp_start: float,
    egovlp_end: float,
    fps: float,
    window: WindowSpec,
) -> Optional[LocalizationResult]:
    """
    Coarse localization with chain-of-thought reasoning.
    """
    print(f"[Coarse w/ reasoning] Analyzing at {fps} fps with detailed reasoning...")

    prompt = (
        f'Event: "{query_text}"\n'
        f"EgoVLP prediction: {egovlp_start:.2f}-{egovlp_end:.2f} seconds.\n\n"
        "Analyze this video step-by-step:\n"
        "1. Describe what you see overall in the video\n"
        "2. Identify moments that could match the event description\n"
        "3. For each potential match, explain why it does or doesn't fit\n"
        "4. Determine the most accurate start and end times\n"
        "5. Rate your confidence and explain any uncertainties\n\n"
        "Provide your step-by-step analysis, then output JSON:\n"
        '{"present": true/false, "start": <float>, "end": <float>, '
        '"confidence": <0-1>, "rationale": "brief summary"}\n'
        "Times are in seconds from video start."
    )

    raw_output = _run_qwen_full_video(
        model=model,
        processor=processor,
        video_path=video_path,
        prompt_text=prompt,
        fps=fps,
        max_tokens=512,  # More tokens for reasoning
        do_sample=False,
    )

    result = parse_localization_output(raw_output, window)

    if result and result.reasoning_chain:
        print(f"  Reasoning: {result.reasoning_chain[:200]}...")

    return result


# =========================
# Adaptive refinement
# =========================


def adaptive_refinement(
    model,
    processor,
    video_path: str,
    query_text: str,
    coarse_result: LocalizationResult,
    egovlp_start: float,
    egovlp_end: float,
    window: WindowSpec,
    max_drift: float = 2.0,
) -> Optional[LocalizationResult]:
    """
    Refine with FPS adapted to coarse confidence.
    Lower confidence = higher FPS for more careful analysis.
    """
    coarse_conf = coarse_result.confidence or 0.5

    # Adaptive FPS: lower confidence = higher FPS
    if coarse_conf > 0.8:
        refine_fps = 4.0
        print(
            f"[Refine] High confidence ({coarse_conf:.2f}), using moderate FPS: {refine_fps}"
        )
    elif coarse_conf > 0.5:
        refine_fps = 6.0
        print(
            f"[Refine] Medium confidence ({coarse_conf:.2f}), using higher FPS: {refine_fps}"
        )
    else:
        refine_fps = 8.0
        print(
            f"[Refine] Low confidence ({coarse_conf:.2f}), using high FPS: {refine_fps}"
        )

    prompt = (
        f'Event: "{query_text}"\n'
        f"Previous analysis found the event at {coarse_result.start:.2f}-{coarse_result.end:.2f}s "
        f"(confidence: {coarse_conf:.2f}).\n"
        f"Original EgoVLP: {egovlp_start:.2f}-{egovlp_end:.2f}s.\n\n"
        "Refine the temporal boundaries with frame-level precision.\n"
        "Focus especially on the exact moment the event starts and ends.\n"
        "Make small adjustments unless you have strong visual evidence for larger changes.\n\n"
        "Output ONLY JSON:\n"
        '{"start": <float>, "end": <float>, "confidence": <0-1>, '
        '"refinement_notes": "what changed and why"}\n'
        "Times in seconds from video start."
    )

    raw_output = _run_qwen_full_video(
        model=model,
        processor=processor,
        video_path=video_path,
        prompt_text=prompt,
        fps=refine_fps,
        max_tokens=256,
        do_sample=False,
    )

    result = parse_localization_output(
        raw_output, window, assume_present_if_missing=True
    )

    if result:
        # Clamp refinement drift
        if max_drift:
            result.start = max(
                coarse_result.start - max_drift,
                min(result.start, coarse_result.start + max_drift),
            )
            result.end = max(
                result.start,
                max(
                    coarse_result.end - max_drift,
                    min(result.end, coarse_result.end + max_drift),
                ),
            )

        result.is_refined = True
        print(f"  Refined to: [{result.start:.2f}, {result.end:.2f}]")

    return result


# =========================
# Validation
# =========================


def validate_final_result(
    model,
    processor,
    video_path: str,
    query_text: str,
    final_start: float,
    final_end: float,
    fps: float = 4.0,
) -> Dict[str, Any]:
    """
    Final validation pass with yes/no verification.
    """
    print(f"[Validation] Verifying result [{final_start:.2f}, {final_end:.2f}]...")

    prompt = (
        f'Event: "{query_text}"\n'
        f"Proposed time range: {final_start:.2f}-{final_end:.2f} seconds.\n\n"
        "Watch the video carefully, especially the proposed time range.\n"
        "Does the described event actually occur during this time range?\n\n"
        "Output ONLY JSON:\n"
        '{"occurs_in_range": true/false, '
        '"confidence": <0-1>, '
        '"issues": "any problems or concerns, or empty string"}\n'
    )

    raw_output = _run_qwen_full_video(
        model=model,
        processor=processor,
        video_path=video_path,
        prompt_text=prompt,
        fps=fps,
        max_tokens=256,
        do_sample=False,
    )

    obj = _extract_json_block(raw_output)

    if obj:
        occurs = obj.get("occurs_in_range", True)
        confidence = obj.get("confidence", 0.5)
        issues = obj.get("issues", "")

        print(
            f"  Validation: {'PASS' if occurs else 'FAIL'}, "
            f"confidence={confidence:.2f}"
        )
        if issues:
            print(f"  Issues noted: {issues}")

        return {
            "validated": occurs,
            "confidence": confidence,
            "issues": issues,
            "raw": raw_output,
        }

    return {"validated": True, "confidence": 0.5, "issues": "", "raw": raw_output}


# =========================
# Main enhanced API
# =========================


def refine_temporal_boundaries_from_egovlp(
    video_path: str,
    query_text: str,
    egovlp_start: float,
    egovlp_end: float,
    video_duration: Optional[float] = None,
    coarse_fps: float = 2.0,
    refine_fps: float = 4.0,
    max_drift_coarse: float = 3.0,
    max_drift_refine: float = 2.0,
    use_multi_hypothesis: bool = True,
    use_grounded_tracking: bool = True,
    use_chain_of_thought: bool = True,
    use_validation: bool = True,
    n_hypotheses: int = 3,
):
    """
    Enhanced temporal boundary refinement with multiple strategies.

    Args:
        video_path: Path to video file
        query_text: Event description
        egovlp_start: EgoVLP predicted start time (seconds)
        egovlp_end: EgoVLP predicted end time (seconds)
        video_duration: Total video duration (optional)
        coarse_fps: FPS for coarse pass
        refine_fps: Base FPS for refinement (may be adjusted adaptively)
        max_drift_coarse: Max deviation from EgoVLP in coarse pass
        max_drift_refine: Max deviation from coarse in refine pass
        use_multi_hypothesis: Generate multiple predictions and aggregate
        use_grounded_tracking: Use object/action tracking
        use_chain_of_thought: Enable detailed reasoning in coarse pass
        use_validation: Validate final result
        n_hypotheses: Number of hypotheses for multi-hypothesis mode

    Returns:
        Dictionary with egovlp, coarse, refined, tracking, validation results
    """
    assert egovlp_end >= egovlp_start, "egovlp_end must be >= egovlp_start"

    print("=" * 60)
    print("ENHANCED TEMPORAL LOCALIZATION")
    print(f"Video: {video_path}")
    print(f"Query: {query_text}")
    print(f"EgoVLP prior: [{egovlp_start:.2f}, {egovlp_end:.2f}]")
    print("=" * 60)

    model, processor = _load_qwen3vl(query_text)

    if video_duration is not None:
        full_window = WindowSpec(
            start=0.0, end=video_duration, half_span=video_duration / 2.0
        )
    else:
        full_window = WindowSpec(start=0.0, end=0.0, half_span=0.0)

    results = {
        "egovlp": {"start": egovlp_start, "end": egovlp_end},
        "coarse": None,
        "tracking": None,
        "multi_hypothesis": None,
        "refined": None,
        "validation": None,
    }

    # ----- Strategy 1: Coarse with optional chain-of-thought -----
    if use_chain_of_thought:
        coarse_result = coarse_with_reasoning(
            model,
            processor,
            video_path,
            query_text,
            egovlp_start,
            egovlp_end,
            coarse_fps,
            full_window,
        )
    else:
        # Standard coarse pass (original implementation)
        prompt = (
            f'Event: "{query_text}"\n'
            f"EgoVLP prediction: {egovlp_start:.2f}-{egovlp_end:.2f}s.\n"
            "Watch the video and determine when this event occurs.\n"
            "Output ONLY JSON: "
            '{"present": true/false, "start": <float>, "end": <float>, '
            '"confidence": <0-1>, "rationale": "..."}'
        )
        raw = _run_qwen_full_video(
            model, processor, video_path, prompt, coarse_fps, max_tokens=256
        )
        coarse_result = parse_localization_output(raw, full_window)

    if not coarse_result or not coarse_result.present:
        print("[Coarse] Event not detected, stopping.")
        results["coarse"] = coarse_result
        return results

    # Clamp coarse drift
    if max_drift_coarse:
        coarse_result.start = max(
            0.0,
            max(
                egovlp_start - max_drift_coarse,
                min(coarse_result.start, egovlp_start + max_drift_coarse),
            ),
        )
        coarse_result.end = max(
            coarse_result.start,
            min(
                egovlp_end + max_drift_coarse,
                max(coarse_result.end, egovlp_end - max_drift_coarse),
            ),
        )

    results["coarse"] = coarse_result
    print(
        f"[Coarse] Result: [{coarse_result.start:.2f}, {coarse_result.end:.2f}], "
        f"conf={coarse_result.confidence}"
    )

    # ----- Strategy 2: Grounded tracking (parallel) -----
    tracking_result = None
    if use_grounded_tracking:
        tracking_result = grounded_localization_with_tracking(
            model, processor, video_path, query_text, coarse_fps, full_window
        )
        results["tracking"] = tracking_result

    # ----- Strategy 3: Multi-hypothesis (parallel) -----
    multi_hyp_result = None
    if use_multi_hypothesis and n_hypotheses > 1:
        hypotheses = get_multiple_hypotheses(
            model,
            processor,
            video_path,
            query_text,
            egovlp_start,
            egovlp_end,
            coarse_fps,
            full_window,
            n_hypotheses,
        )

        if hypotheses:
            multi_hyp_result = aggregate_hypotheses(
                hypotheses, method="weighted_median"
            )
            results["multi_hypothesis"] = multi_hyp_result
            print(
                f"[Multi-hyp] Aggregated from {len(hypotheses)} predictions: "
                f"[{multi_hyp_result.start:.2f}, {multi_hyp_result.end:.2f}]"
            )

    # ----- Choose best coarse estimate -----
    # Prioritize based on confidence or use ensemble
    candidates = [
        ("coarse", coarse_result),
        ("tracking", tracking_result),
        ("multi_hypothesis", multi_hyp_result),
    ]

    valid_candidates = [
        (name, res)
        for name, res in candidates
        if res and res.present and res.confidence
    ]

    if valid_candidates:
        best_name, best_coarse = max(valid_candidates, key=lambda x: x[1].confidence)
        print(f"[Selection] Using '{best_name}' for refinement (highest confidence)")
    else:
        best_coarse = coarse_result
        print("[Selection] Using standard coarse for refinement")

    # ----- Adaptive refinement -----
    refined_result = adaptive_refinement(
        model,
        processor,
        video_path,
        query_text,
        best_coarse,
        egovlp_start,
        egovlp_end,
        full_window,
        max_drift_refine,
    )

    results["refined"] = refined_result

    # ----- Validation -----
    if use_validation and refined_result:
        validation = validate_final_result(
            model,
            processor,
            video_path,
            query_text,
            refined_result.start,
            refined_result.end,
        )
        results["validation"] = validation

        if not validation["validated"] and validation["confidence"] > 0.7:
            print("[WARNING] Validation suggests result may be incorrect!")

    return results


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # Example: ego4d 15 s clip
    video_file = (
        "/dais/fs/scratch/dduka/databases/ego4d/video_320px_15sec/"
        "/9e4edf4d-e557-4b3d-bc35-0d7f1f91019b.mp4/0.mp4"
    )

    for s, e, q in [(0.0, 8.0, "#C C carries a pot from the cooker"), (3.0, 5.5, "#C C drops the pot on the cooker"), (4.0, 9.0, "#C C carries a sauce pan from the counter")]:

        search_query = q

        # EgoVLP prediction in seconds
        egovlp_start = s
        egovlp_end = e

        video_duration_sec = 15.0

        result = refine_temporal_boundaries_from_egovlp(
            video_path=video_file,
            query_text=search_query,
            egovlp_start=egovlp_start,
            egovlp_end=egovlp_end,
            video_duration=video_duration_sec,
            coarse_fps=15.0,
            refine_fps=30.0,
            max_drift_coarse=3.0,
            max_drift_refine=2.0,
            use_multi_hypothesis=True,
            use_grounded_tracking=True,
            use_chain_of_thought=True,
            use_validation=True,
            n_hypotheses=5,
        )

        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(
            f"EgoVLP:      [{result['egovlp']['start']:.2f}, {result['egovlp']['end']:.2f}]"
        )

        if result["coarse"]:
            c = result["coarse"]
            print(f"Coarse:      [{c.start:.2f}, {c.end:.2f}] (conf={c.confidence:.2f})")

        if result["tracking"]:
            t = result["tracking"]
            print(f"Tracking:    [{t.start:.2f}, {t.end:.2f}] (conf={t.confidence:.2f})")

        if result["multi_hypothesis"]:
            m = result["multi_hypothesis"]
            print(
                f"Multi-hyp:   [{m.start:.2f}, {m.end:.2f}] (conf={m.confidence:.2f}, "
                f"n={m.hypotheses_count})"
            )

        if result["refined"]:
            r = result["refined"]
            print(f"REFINED:     [{r.start:.2f}, {r.end:.2f}]")

        if result["validation"]:
            print(f"Query: {q}")
            v = result["validation"]
            status = "✓ VALIDATED" if v["validated"] else "✗ VALIDATION FAILED"
            print(f"Validation:  {status} (conf={v['confidence']:.2f})")
            if v["issues"]:
                print(f"  Issues: {v['issues']}")

        print("=" * 60)
