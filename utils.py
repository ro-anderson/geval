# utils.py
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _logsumexp(values: Iterable[float]) -> float:
    """Stable log-sum-exp for a list/iterable of log-probabilities."""
    vals = list(values)
    if not vals:
        return float('-inf')
    m = max(vals)
    if m == float('-inf'):
        return float('-inf')
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _score_range(min_score: int, max_score: int) -> Tuple[int, ...]:
    if min_score > max_score:
        raise ValueError("min_score must be <= max_score")
    return tuple(range(int(min_score), int(max_score) + 1))


def _allowed_digit_tokens(min_score: int, max_score: int) -> set:
    """
    Allowed single-token numeric strings that we expect to see at the score position.
    For 0..10 or 1..10 this will be {'0','1',...,'9','10'}.
    Note: numbers > 10 may tokenize into multiple tokens; we do not try to reconstruct them here.
    """
    return {str(s) for s in _score_range(min_score, max_score) if s <= 10}


def normalize_logprob_dict(
    logprob_by_score: Mapping[int, float],
    min_score: int,
    max_score: int,
) -> Dict[int, float]:
    """
    Convert a (possibly partial) dict {score: logprob} into a full probability distribution
    over [min_score..max_score].
    """
    full_keys = _score_range(min_score, max_score)
    dense: Dict[int, float] = {k: float('-inf') for k in full_keys}
    for k, v in logprob_by_score.items():
        if k in dense:
            dense[k] = float(v)

    lse = _logsumexp(dense.values())
    if lse == float('-inf'):
        # Degenerate: everything is -inf; fall back to uniform
        n = len(full_keys)
        return {k: 1.0 / n for k in full_keys}

    return {k: math.exp(v - lse) for k, v in dense.items()}


def _as_attr(obj: Any, name: str, default=None):
    """Get attribute or dict item with a single helper."""
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _strip_token(tok: Any) -> str:
    s = tok if isinstance(tok, str) else _as_attr(tok, "token", "") or ""
    return s.strip()


def extract_score_token_indexed_distribution(
    choice: Any,
    min_score: int,
    max_score: int,
    anchor_key: str = '"score"',
) -> Dict[int, float]:
    """
    Find the numeric token(s) that serve as the value for the JSON key indicated by `anchor_key`
    (default: '"score"') and build a {score: logprob} map from top_logprobs at that position.

    Special handling for the 2-digit "10":
    - If "10" appears in the top_logprobs at the first numeric position, use that logprob.
    - Else, if "1" is present at the first position and the next token's top_logprobs contain "0",
      approximate logprob("10") as logprob("1") + logprob("0").

    Returns natural-log probabilities (not normalized). If nothing is found, returns {}.
    """
    allowed_tokens = _allowed_digit_tokens(min_score, max_score)
    content_items: List[Any] = _as_attr(_as_attr(choice, "logprobs"), "content", []) or []

    # Build a simple token stream so we can detect the sequence [anchor_key, ':', <number>]
    tokens = [(_strip_token(it), it) for it in content_items]

    i = 0
    while i < len(tokens):
        tok_i, item_i = tokens[i]
        if tok_i.strip() == anchor_key:
            # seek colon
            j = i + 1
            while j < len(tokens) and tokens[j][0].strip() in {"", " ", "\n"}:
                j += 1
            if j < len(tokens) and tokens[j][0].strip() == ":":
                # seek the first non-space token after colon
                k = j + 1
                while k < len(tokens) and tokens[k][0].strip() in {"", " ", "\n"}:
                    k += 1
                if k < len(tokens):
                    tok_k, item_k = tokens[k]
                    digit = tok_k.strip()
                    if digit and (digit[0] in "0123456789"):
                        # collect top_logprobs at this position
                        lp_map: Dict[int, float] = {}
                        top_k = _as_attr(item_k, "top_logprobs", []) or []
                        for cand in top_k:
                            ctok = _strip_token(cand)
                            if ctok in allowed_tokens:
                                s_int = int(ctok)
                                lp_map[s_int] = float(_as_attr(cand, "logprob", float("-inf")))

                        # ensure the chosen token itself is present
                        chosen_lp = float(_as_attr(item_k, "logprob", float("-inf")))
                        if digit in allowed_tokens:
                            lp_map.setdefault(int(digit), chosen_lp)

                        # Special-case combine for "10" if needed and in range
                        if max_score >= 10 and 10 >= int(min_score) and 10 <= int(max_score):
                            if 10 not in lp_map:
                                # approx: log P("10") ~= log P("1") + log P("0" at next pos)
                                lp_one = None
                                if "1" in allowed_tokens:
                                    if 1 in lp_map:
                                        lp_one = lp_map[1]
                                    elif digit == "1":
                                        lp_one = chosen_lp
                                lp_zero_next = None
                                if k + 1 < len(tokens):
                                    tok_next, item_next = tokens[k + 1]
                                    top_next = _as_attr(item_next, "top_logprobs", []) or []
                                    for cand2 in top_next:
                                        if _strip_token(cand2) == "0":
                                            lp_zero_next = float(_as_attr(cand2, "logprob", float("-inf")))
                                            break
                                    if lp_zero_next is None and tok_next.strip() == "0":
                                        lp_zero_next = float(_as_attr(item_next, "logprob", float("-inf")))

                                if lp_one is not None and lp_zero_next is not None:
                                    lp_map[10] = float(lp_one + lp_zero_next)

                        # filter lp_map to valid score range
                        valid_lp = {s: lp for s, lp in lp_map.items() if min_score <= s <= max_score}
                        return valid_lp
        i += 1

    return {}  # not found


def score_distribution_from_choice(
    choice: Any,
    min_score: int = 1,
    max_score: int = 5,
    anchor_key: str = '"score"',
) -> Dict[int, float]:
    """
    Construct P(score=s | choice) for s in [min_score..max_score] from top_logprobs at the anchored score token.
    If we can't find a numeric token, fall back to uniform.
    """
    lp_map = extract_score_token_indexed_distribution(choice, min_score, max_score, anchor_key)
    if not lp_map:
        rng = _score_range(min_score, max_score)
        return {k: 1.0 / len(rng) for k in rng}
    return normalize_logprob_dict(lp_map, min_score, max_score)


def expected_score(
    distribution: Mapping[int, float],
    min_score: int = 1,
    max_score: int = 5,
) -> float:
    """Compute E[score] = sum_s s * P(s)."""
    return float(sum(s * float(distribution.get(s, 0.0)) for s in range(min_score, max_score + 1)))


def expected_score01(
    distribution: Mapping[int, float],
    min_score: int,
    max_score: int,
) -> float:
    """Compute expected score normalized to [0,1]."""
    if max_score == min_score:
        return 0.0
    ev = expected_score(distribution, min_score, max_score)
    return (ev - min_score) / (max_score - min_score)


def prob_of_score(distribution: Mapping[int, float], score: int) -> float:
    """Return P(score) given a distribution (0 if missing)."""
    return float(distribution.get(int(score), 0.0))


def summarize_choice(
    choice: Any,
    parsed_score: int,
    min_score: int = 1,
    max_score: int = 5,
    dist_override: Optional[Mapping[int, float]] = None,
    parsed_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Package everything we typically want to store for a single choice.
    If dist_override is given, use it instead of recomputing from the choice.
    """
    dist = dict(dist_override) if dist_override is not None else score_distribution_from_choice(
        choice, min_score=min_score, max_score=max_score
    )
    p_emitted = prob_of_score(dist, parsed_score)
    ev = expected_score(dist, min_score=min_score, max_score=max_score)
    ev01 = expected_score01(dist, min_score=min_score, max_score=max_score)

    return {
        "score": int(parsed_score),
        "distribution": {int(k): float(v) for k, v in dist.items()},
        "p_of_score": float(p_emitted),
        "normalized_score": float(ev),
        "normalized_score_01": float(ev01),
        "reason": parsed_reason,
    }


# Token usage tracking utilities

def extract_token_usage(response: Any) -> Dict[str, int]:
    """
    Extract token usage information from an OpenAI API response.
    Returns a dictionary with token counts or zeros if not available.
    """
    if hasattr(response, 'usage') and response.usage:
        return {
            'total_tokens': response.usage.total_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens
        }
    else:
        return {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0
        }


def accumulate_token_usage(
    current_totals: Dict[str, int], 
    new_usage: Dict[str, int]
) -> Dict[str, int]:
    """
    Accumulate token usage from a new response into running totals.
    """
    return {
        'total_tokens': current_totals.get('total_tokens', 0) + new_usage.get('total_tokens', 0),
        'prompt_tokens': current_totals.get('prompt_tokens', 0) + new_usage.get('prompt_tokens', 0),
        'completion_tokens': current_totals.get('completion_tokens', 0) + new_usage.get('completion_tokens', 0)
    }


def print_token_usage_summary(
    token_totals: Dict[str, int], 
    instances_processed: int, 
    instances_ignored: int = 0
) -> None:
    """
    Print a comprehensive summary of token usage and costs.
    """
    print(f'Ignored total: {instances_ignored}')
    print(f'Total instances processed: {instances_processed}')
    print(f'Total tokens used: {token_totals["total_tokens"]:,}')
    print(f'Total prompt tokens: {token_totals["prompt_tokens"]:,}')
    print(f'Total completion tokens: {token_totals["completion_tokens"]:,}')
    
    if instances_processed > 0:
        avg_tokens = token_totals["total_tokens"] / instances_processed
        print(f'Average tokens per instance: {avg_tokens:.1f}')
    else:
        print('Average tokens per instance: 0.0')


def count_positive_integers_in_range(min_val: int, max_val: int) -> int:
    """
    Return the number of strictly positive integers in the inclusive range [min_val, max_val].

    Constraints:
    - min_val and max_val must be integers
    - 0 <= min_val < max_val  (equality is NOT allowed)
    """
    if not isinstance(min_val, int) or not isinstance(max_val, int):
        raise TypeError("min and max must be integers")
    if min_val < 0 or max_val < 0:
        raise ValueError("min and max must be >= 0")
    if min_val == max_val:
        raise ValueError("min and max cannot be equal")
    if min_val > max_val:
        raise ValueError("min cannot be greater than max")

    start = max(min_val, 1)
    return max(0, max_val - start + 1)