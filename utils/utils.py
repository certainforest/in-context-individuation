import os
import re
import time
import asyncio
import bisect
import requests
import pandas as pd
from dotenv import load_dotenv
from itertools import accumulate
from pathlib import Path


load_dotenv()

def parse_transcript(doc_path, speakers, chunk_size: int | None = None):
    """
    Parse transcript into line-level records, with optional fixed-size chunks.

    A turn is defined as all contiguous lines spoken by the same speaker.
    If a line starts with 'SPEAKER:', that starts a new turn.
    Otherwise, the line is treated as a continuation of the current turn.

    If chunk_size is None (default), returns line-level records:
        Each dict has:
            - line_id
            - turn_id
            - turn_speaker
            - line_text

    If chunk_size is set (e.g., 25), returns chunk records:
        Each dict has:
            - source
            - chunk_ix
            - lines
            - speakers
            - speaker_line_count
            - baseline_text
            - unlabeled_text
            - num_turns
    """
    with Path(doc_path).open("r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    lines = []
    current_speaker = None
    current_turn_id = 0
    line_id = 0

    for raw_line in raw_lines:
        matched_speaker = None
        text = raw_line

        for speaker in speakers:
            prefix = f"{speaker}:"
            if raw_line.startswith(prefix):
                matched_speaker = speaker
                text = raw_line[len(prefix):].strip()
                break

        if matched_speaker is not None:
            current_speaker = matched_speaker
            current_turn_id += 1

        if current_speaker is not None:
            line_id += 1
            lines.append({
                "line_id": line_id,
                "turn_id": current_turn_id,
                "turn_speaker": current_speaker,
                "line_text": text,
            })

    if chunk_size is None:
        return lines
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    chunks = []
    doc_stem = Path(doc_path).stem

    for chunk_ix, start_ix in enumerate(range(0, len(lines), chunk_size)):
        chunk_lines = lines[start_ix:start_ix + chunk_size]
        if not chunk_lines:
            continue

        speaker_line_count = {}
        for row in chunk_lines:
            speaker = row["turn_speaker"]
            speaker_line_count[speaker] = speaker_line_count.get(speaker, 0) + 1

        first_line_id = chunk_lines[0]["line_id"]
        last_line_id = chunk_lines[-1]["line_id"]
        speakers_in_chunk = set(speaker_line_count.keys())
        unique_turns = {row["turn_id"] for row in chunk_lines}

        chunks.append(
            {
                "source": f"{doc_stem}:{first_line_id}-{last_line_id}",
                "chunk_ix": chunk_ix,
                "lines": chunk_lines,
                "speakers": speakers_in_chunk,
                "speaker_line_count": speaker_line_count,
                "baseline_text": "\n".join(
                    [f"{row['turn_speaker']}: {row['line_text']}" for row in chunk_lines]
                ) + "\n",
                "unlabeled_text": "\n".join([row["line_text"] for row in chunk_lines]) + "\n",
                "num_turns": len(unique_turns),
            }
        )

    return chunks

def flag_message_types(
    sample_level_df: pd.DataFrame,
    base_messages: list[str],
    allow_ambiguous: bool = False,
) -> pd.DataFrame:
    """
    Take a token-level df with column `token`; identify whether each token is one of
    base_messages, treating the entire dataframe as a single prompt.

    Description:
        High accuracy exact match version
            1) Concatenate tokens (in dataframe row order) -> full_text
            2) For each base_message t in base_messages:
                - find all exact occurrences of t in full_text
                - for each occurrence, mark all tokens whose character spans overlap t
            3) For each token:
                - if it belongs to exactly one base_message -> base_message_ix = that index
                - if it belongs to >1 base_messages       -> base_message_ix = None, unless allow_ambiguous=True
                - else                                    -> base_message_ix = None

        Notes:
            - Tokens are concatenated without separators.
            - Matching is exact and can cross token boundaries.
            - Token order is the current dataframe row order.
            - Be careful to avoid overlapping base_messages unless allow_ambiguous=True.

    Params:
        @sample_level_df: A token-level dataframe with column `token`
        @base_messages: A list of base messages to match tokens to
        @allow_ambiguous: If True, ambiguous tokens are assigned to the first matching
            base_message by index

    Returns:
        The original dataframe, with additional columns:
            - `base_message_ix`
            - `base_message`
    """
    if 'token' not in sample_level_df.columns:
        raise ValueError("sample_level_df must contain a 'token' column")

    df = sample_level_df.reset_index(drop=True).copy()

    # Early exit if nothing to match
    if not base_messages:
        df['base_message_ix'] = None
        df['base_message'] = None
        return df

    n_rows = len(df)
    memberships: list[set[int]] = [set() for _ in range(n_rows)]

    tokens = df['token'].astype(str).tolist()

    # Build full text and token spans
    token_lens = [len(t) for t in tokens]
    token_ends = list(accumulate(token_lens))
    token_starts = [0] + token_ends[:-1]
    full_text = ''.join(tokens)
    last_token_end = token_ends[-1] if token_ends else 0

    # For each base_message, find all exact matches
    for bm_ix, bm in enumerate(base_messages):
        if not bm:
            continue  # skip empty templates

        bm_len = len(bm)
        start_pos = 0

        while (match_at := full_text.find(bm, start_pos)) != -1:
            match_end = match_at + bm_len

            # Optimization: if match starts beyond last token, nothing else to do
            if match_at >= last_token_end:
                break

            # Find overlapping tokens via binary search
            first = bisect.bisect_right(token_ends, match_at)
            last = bisect.bisect_left(token_starts, match_end)

            for token_ix in range(first, last):
                memberships[token_ix].add(bm_ix)

            # Move forward to find additional (possibly overlapping) matches
            start_pos = match_at + 1

    # Convert membership sets into output columns
    base_message_ix_col: list[int | None] = []

    for row_ix, s in enumerate(memberships):
        if len(s) > 1:
            if allow_ambiguous:
                base_message_ix_col.append(min(s))
            else:
                token = df.loc[row_ix, 'token']
                matched = [base_messages[i] for i in s]
                raise ValueError(
                    f"Ambiguous match at row {row_ix}, token {token!r}: "
                    f"matched {len(s)} base_messages: {matched}"
                )
        elif len(s) == 1:
            base_message_ix_col.append(next(iter(s)))
        else:
            base_message_ix_col.append(None)

    df['base_message_ix'] = base_message_ix_col
    df['base_message'] = [
        base_messages[i] if i is not None else None
        for i in base_message_ix_col
    ]

    return df


##### Openrouter functions #####
def send_openrouter_request(
    messages,
    model="google/gemini-2.5-pro",
    provider_order=None,
    allow_fallbacks=True,
    temperature=0.0,
    max_tokens=4000,
):
    """
    Submit a prompt to OpenRouter using requests.
    Returns: (final_response, reasoning, refusal, provider)
    """
    if provider_order is None:
        provider_order = ["deepinfra/fp4", "google-vertex/global", "google-vertex/us"]

    openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "gptoss-jailbreak-evals",
        "HTTP-Referer": "https://localhost",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if provider_order is not None:
        payload["provider"] = {
            "order": provider_order,
            "allow_fallbacks": allow_fallbacks,
        }

    for attempt in range(3):
        try:
            r = requests.post(openrouter_url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            message = data["choices"][0]["message"]
            final_response = message.get("content")
            reasoning = message.get("reasoning")
            refusal = message.get("refusal")
            provider = data.get("provider")
            return final_response, reasoning, refusal, provider
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise e

async def send_openrouter_request_async(
    messages,
    model="google/gemini-2.5-pro",
    provider_order=None,
    allow_fallbacks=True,
    temperature=0.0,
    max_tokens=4000,
):
    """
    Async wrapper for send_openrouter_request.
    """
    return await asyncio.to_thread(
        send_openrouter_request,
        messages=messages,
        model=model,
        provider_order=provider_order,
        allow_fallbacks=allow_fallbacks,
        temperature=temperature,
        max_tokens=max_tokens,
    )

## Restyling (for variants)
def restyle_variant_text(
    transcript_text: str,
    instruction: str,
    model: str = "google/gemini-2.5-pro",
    temperature: float = 0.8,
    max_tokens: int = 4000,
) -> tuple[str, str | None]:
    """
    Restyle a transcript using a single free-form instruction.

    Args:
        transcript_text: Full transcript text.
        instruction: Free-form instruction, e.g.
            "Make all text belonging to speaker PINK all caps."
        model: OpenRouter model id.
        temperature: Sampling temperature.
        max_tokens: Max generated tokens.

    Returns:
        Tuple of (restyled_transcript, reasoning).
    """
    if not transcript_text or not transcript_text.strip():
        raise ValueError("transcript_text must be a non-empty string")
    if not instruction or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")

    system_prompt = (
        "You rewrite transcripts. "
        "Return only the transformed transcript text with no commentary."
    )
    user_prompt = f"""
Instruction:
{instruction}

Transcript:
{transcript_text}
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response, reason, _, _ = send_openrouter_request(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.strip(), reason
