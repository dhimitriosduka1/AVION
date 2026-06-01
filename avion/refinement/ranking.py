import copy
import math

from avion.refinement.candidates import anchor_distance, temporal_iou
from avion.refinement.io import as_float


MODEL_NAMES = ("lavila", "egovlp")
NEG_INF = -1.0e9


def finite_float(value):
    value = as_float(value)
    if value is None or not math.isfinite(value):
        return None
    return value


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def z_normalize(values, eps=1e-8):
    finite = [
        value
        for value in values
        if value is not None and math.isfinite(value) and value > NEG_INF / 10.0
    ]
    if not finite:
        return [0.0 for _ in values]
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / len(finite)
    std = math.sqrt(var)
    if std < eps:
        return [0.0 for _ in values]
    return [
        0.0
        if value is None or not math.isfinite(value) or value <= NEG_INF / 10.0
        else (value - mean) / std
        for value in values
    ]


def softmax(scores, temperature=1.0):
    if temperature <= 0:
        temperature = 1.0
    finite_scores = [score for score in scores if math.isfinite(score)]
    if not scores:
        return []
    if not finite_scores:
        return [1.0 / len(scores) for _ in scores]
    max_score = max(finite_scores)
    exps = []
    for score in scores:
        if not math.isfinite(score):
            exps.append(0.0)
        else:
            exps.append(math.exp((score - max_score) / temperature))
    denom = sum(exps)
    if denom <= 0:
        return [1.0 / len(scores) for _ in scores]
    return [value / denom for value in exps]


def weighted_mean(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))


def weighted_std(values, weights):
    if not values:
        return 0.0
    mean = weighted_mean(values, weights)
    var = sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights))
    return math.sqrt(max(0.0, var))


def distribution_stats(candidates, scores, temperature):
    probs = softmax(scores, temperature)
    starts = [candidate["start_sec"] for candidate in candidates]
    ends = [candidate["end_sec"] for candidate in candidates]
    entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
    return {
        "probabilities": probs,
        "ranking_entropy": entropy,
        "start_uncertainty": weighted_std(starts, probs),
        "end_uncertainty": weighted_std(ends, probs),
    }


def model_has_scores(candidates, model_name):
    key = f"{model_name}_sim_current"
    return any(finite_float(candidate.get(key)) is not None for candidate in candidates)


def available_models(candidates, requested_mode):
    present = [model for model in MODEL_NAMES if model_has_scores(candidates, model)]
    if requested_mode == "lavila_only":
        return ["lavila"] if "lavila" in present else []
    if requested_mode == "egovlp_only":
        return ["egovlp"] if "egovlp" in present else []
    return present


def actual_ranking_mode(models):
    if models == ["lavila"]:
        return "lavila_only"
    if models == ["egovlp"]:
        return "egovlp_only"
    if set(models) == {"lavila", "egovlp"}:
        return "lavila_egovlp_ensemble"
    return "none"


def compute_per_model_scores(candidates, record, model_name, ranking_cfg):
    current = [finite_float(candidate.get(f"{model_name}_sim_current")) for candidate in candidates]
    previous = [finite_float(candidate.get(f"{model_name}_sim_prev")) for candidate in candidates]
    next_ = [finite_float(candidate.get(f"{model_name}_sim_next")) for candidate in candidates]
    if all(value is None for value in current):
        return None

    current_z = z_normalize(current)
    previous_z = z_normalize(previous) if any(value is not None for value in previous) else None
    next_z = z_normalize(next_) if any(value is not None for value in next_) else None

    qwen_start = as_float(record.get("qwen_start_sec"), candidates[0]["start_sec"])
    qwen_end = as_float(record.get("qwen_end_sec"), candidates[0]["end_sec"])
    timestamp = as_float(record.get("timestamp_sec"), 0.5 * (qwen_start + qwen_end))
    preferred_max = ranking_cfg["preferred_max_duration"]

    raw_scores = []
    neighbor_separations = []
    for idx, candidate in enumerate(candidates):
        if current[idx] is None:
            raw_scores.append(NEG_INF)
            neighbor_separations.append(0.0)
            continue

        neighbor_values = []
        if previous_z is not None:
            neighbor_values.append(previous_z[idx])
        if next_z is not None:
            neighbor_values.append(next_z[idx])
        neighbor_score = max(neighbor_values) if neighbor_values else 0.0

        duration = candidate["end_sec"] - candidate["start_sec"]
        length_penalty = max(0.0, duration - preferred_max) / preferred_max
        qwen_penalty = abs(candidate["start_sec"] - qwen_start) + abs(candidate["end_sec"] - qwen_end)
        anchor_penalty = anchor_distance(candidate["start_sec"], candidate["end_sec"], timestamp)

        score = (
            current_z[idx]
            - ranking_cfg["neighbor_weight"] * neighbor_score
            - ranking_cfg["length_weight"] * length_penalty
            - ranking_cfg["qwen_prior_weight"] * qwen_penalty
            - ranking_cfg["anchor_weight"] * anchor_penalty
        )
        raw_scores.append(score)
        if neighbor_values:
            neighbor_separations.append(current_z[idx] - neighbor_score)
        else:
            neighbor_separations.append(0.0)

    normalized_scores = z_normalize(raw_scores)
    normalized_scores = [
        NEG_INF if current[idx] is None else value
        for idx, value in enumerate(normalized_scores)
    ]
    for idx, candidate in enumerate(candidates):
        candidate[f"{model_name}_ranking_score"] = raw_scores[idx]
        candidate[f"{model_name}_normalized_score"] = normalized_scores[idx]
        candidate[f"{model_name}_neighbor_separation"] = neighbor_separations[idx]

    best_idx = max(range(len(candidates)), key=lambda idx: normalized_scores[idx])
    return {
        "raw_scores": raw_scores,
        "normalized_scores": normalized_scores,
        "neighbor_separations": neighbor_separations,
        "best_idx": best_idx,
    }


def find_closest_candidate(candidates, start, end):
    if start is None or end is None or not candidates:
        return None
    return min(
        range(len(candidates)),
        key=lambda idx: abs(candidates[idx]["start_sec"] - start) + abs(candidates[idx]["end_sec"] - end),
    )


def score_margin(sorted_scores):
    if not sorted_scores:
        return 0.0
    if len(sorted_scores) == 1:
        return 0.0
    return sorted_scores[0] - sorted_scores[1]


def confidence_score(score_margin_value, neighbor_separation, agreement_iou, entropy, start_uncertainty, end_uncertainty, cfg):
    value = (
        cfg["margin_weight"] * score_margin_value
        + cfg["neighbor_weight"] * neighbor_separation
        - cfg["entropy_weight"] * entropy
        - cfg["start_uncertainty_weight"] * start_uncertainty
        - cfg["end_uncertainty_weight"] * end_uncertainty
    )
    if agreement_iou is not None:
        value += cfg["agreement_weight"] * agreement_iou
    return sigmoid(value)


def rank_scored_record(scored_record, config):
    record = scored_record.get("record") or scored_record
    candidates = copy.deepcopy(scored_record.get("candidates") or [])
    if not candidates:
        return fallback_result(record, "too_few_valid_candidates"), []

    requested_mode = config.get("mode", "lavila_egovlp_ensemble")
    models = available_models(candidates, requested_mode)
    if not models:
        return fallback_result(record, "missing_model_scores"), []

    ranking_cfg = config["ranking"]
    per_model = {}
    for model_name in models:
        model_scores = compute_per_model_scores(candidates, record, model_name, ranking_cfg)
        if model_scores is not None:
            per_model[model_name] = model_scores

    models = [model for model in models if model in per_model]
    if not models:
        return fallback_result(record, "missing_model_scores"), []

    lavila_best = None
    egovlp_best = None
    if "lavila" in per_model:
        lavila_best = candidates[per_model["lavila"]["best_idx"]]
    if "egovlp" in per_model:
        egovlp_best = candidates[per_model["egovlp"]["best_idx"]]

    final_scores = []
    if len(models) == 1:
        model_name = models[0]
        final_scores = list(per_model[model_name]["normalized_scores"])
    else:
        for candidate in candidates:
            lavila_score = candidate.get("lavila_normalized_score", 0.0)
            egovlp_score = candidate.get("egovlp_normalized_score", 0.0)
            agreement_bonus = 0.5 * temporal_iou(
                (candidate["start_sec"], candidate["end_sec"]),
                (lavila_best["start_sec"], lavila_best["end_sec"]),
            ) + 0.5 * temporal_iou(
                (candidate["start_sec"], candidate["end_sec"]),
                (egovlp_best["start_sec"], egovlp_best["end_sec"]),
            )
            final_scores.append(
                ranking_cfg["lavila_weight"] * lavila_score
                + ranking_cfg["egovlp_weight"] * egovlp_score
                + ranking_cfg["agreement_bonus_weight"] * agreement_bonus
            )

    for idx, candidate in enumerate(candidates):
        candidate["ensemble_score"] = final_scores[idx]

    ranked_indices = sorted(range(len(candidates)), key=lambda idx: final_scores[idx], reverse=True)
    best_idx = ranked_indices[0]
    qwen_idx = find_closest_candidate(
        candidates,
        as_float(record.get("qwen_start_sec")),
        as_float(record.get("qwen_end_sec")),
    )

    selected_idx = best_idx
    update_reason = "updated"
    if qwen_idx is not None:
        best_minus_qwen = final_scores[best_idx] - final_scores[qwen_idx]
        if best_idx == qwen_idx:
            update_reason = "qwen_already_best"
        elif ranking_cfg.get("keep_qwen_if_uncertain", True) and best_minus_qwen < ranking_cfg["min_score_margin_to_update"]:
            selected_idx = qwen_idx
            update_reason = "kept_qwen_uncertain"
    else:
        best_minus_qwen = None

    sorted_scores = [final_scores[idx] for idx in ranked_indices]
    margin = score_margin(sorted_scores)
    dist = distribution_stats(candidates, final_scores, ranking_cfg["temperature"])

    selected = candidates[selected_idx]
    neighbor_values = [
        selected.get(f"{model_name}_neighbor_separation", 0.0)
        for model_name in models
    ]
    neighbor_separation = sum(neighbor_values) / len(neighbor_values) if neighbor_values else 0.0

    agreement_iou = None
    if lavila_best is not None and egovlp_best is not None:
        agreement_iou = temporal_iou(
            (lavila_best["start_sec"], lavila_best["end_sec"]),
            (egovlp_best["start_sec"], egovlp_best["end_sec"]),
        )

    confidence = confidence_score(
        margin,
        neighbor_separation,
        agreement_iou,
        dist["ranking_entropy"],
        dist["start_uncertainty"],
        dist["end_uncertainty"],
        config["confidence"],
    )

    result = {
        "refined_start_sec": selected["start_sec"],
        "refined_end_sec": selected["end_sec"],
        "ranking_confidence": confidence,
        "start_uncertainty": dist["start_uncertainty"],
        "end_uncertainty": dist["end_uncertainty"],
        "score_margin": margin,
        "neighbor_separation": neighbor_separation,
        "lavila_best_start_sec": lavila_best["start_sec"] if lavila_best else None,
        "lavila_best_end_sec": lavila_best["end_sec"] if lavila_best else None,
        "egovlp_best_start_sec": egovlp_best["start_sec"] if egovlp_best else None,
        "egovlp_best_end_sec": egovlp_best["end_sec"] if egovlp_best else None,
        "model_agreement_iou": agreement_iou,
        "ranking_mode": actual_ranking_mode(models),
        "refinement_iteration": config.get("refinement_iteration", 1),
        "ranking_entropy": dist["ranking_entropy"],
        "selected_candidate_id": selected.get("candidate_id"),
        "best_candidate_id": candidates[best_idx].get("candidate_id"),
        "qwen_candidate_id": candidates[qwen_idx].get("candidate_id") if qwen_idx is not None else None,
        "best_minus_qwen_score": best_minus_qwen,
        "update_reason": update_reason,
        "refinement_status": "ok",
    }

    top_k = int(config["output"].get("top_k_candidates", 10))
    ranking_rows = []
    for rank, idx in enumerate(ranked_indices[:top_k], 1):
        candidate = copy.deepcopy(candidates[idx])
        candidate["rank"] = rank
        candidate["probability"] = dist["probabilities"][idx]
        ranking_rows.append(candidate)
    return result, ranking_rows


def fallback_result(record, reason):
    qwen_start = as_float(record.get("qwen_start_sec"))
    qwen_end = as_float(record.get("qwen_end_sec"))
    return {
        "refined_start_sec": qwen_start,
        "refined_end_sec": qwen_end,
        "ranking_confidence": 0.0,
        "start_uncertainty": None,
        "end_uncertainty": None,
        "score_margin": None,
        "neighbor_separation": None,
        "lavila_best_start_sec": None,
        "lavila_best_end_sec": None,
        "egovlp_best_start_sec": None,
        "egovlp_best_end_sec": None,
        "model_agreement_iou": None,
        "ranking_mode": "none",
        "refinement_iteration": None,
        "ranking_entropy": None,
        "selected_candidate_id": None,
        "best_candidate_id": None,
        "qwen_candidate_id": None,
        "best_minus_qwen_score": None,
        "update_reason": reason,
        "refinement_status": "skipped",
        "skip_reason": reason,
    }
