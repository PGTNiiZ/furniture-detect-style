"""Target-style heuristic. Scores are model outputs, not human compatibility labels."""
ALIASES = {"couch": "sofa", "sofa": "sofa", "dining table": "table", "table": "table"}


def canonical_type(value):
    value = value.strip().lower()
    return ALIASES.get(value, value)


def decide(probabilities, target, minimum_confidence=.6, keep_threshold=.6, replace_threshold=.2):
    if target not in probabilities:
        raise ValueError(f"Unsupported target style: {target}")
    if not 0 <= replace_threshold < keep_threshold <= 1:
        raise ValueError("Need 0 <= replace_threshold < keep_threshold <= 1")
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between 0 and 1")
    peak = max(probabilities.values())
    target_score = probabilities[target]
    if peak < minimum_confidence:
        action = "review"
    elif target_score >= keep_threshold:
        action = "keep_candidate"
    elif target_score <= replace_threshold:
        action = "replace_candidate"
    else:
        action = "review"
    return {"action": action, "target_style_score": target_score,
            "reason": "Heuristic from uncalibrated style softmax; human review required"}


def replacements(catalog, furniture_type, target, top_k=3):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    items = [r for r in catalog if canonical_type(r["type"]) == canonical_type(furniture_type)
             and r["style"] == target]
    # ID order is a deterministic baseline, NOT a learned aesthetic ranking.
    return [{"image_id": r["image_id"], "image_path": r["image_path"],
             "type": r["type"], "style": r["style"],
             "reason": "same furniture category and catalog target-style label"}
            for r in sorted(items, key=lambda r: r["image_id"])[:top_k]]
