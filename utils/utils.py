import bisect
from itertools import accumulate
from pathlib import Path
import pandas as pd

def parse_transcript(doc_path, speakers):
    """
    Parse transcript into line-level records.

    A turn is defined as all contiguous lines spoken by the same speaker.
    If a line starts with 'SPEAKER:', that starts a new turn.
    Otherwise, the line is treated as a continuation of the current turn.

    Returns
    -------
    lines : list[dict]
        Each dict has:
        - line_id
        - turn_id
        - turn_speaker
        - line_text
    """
    with Path(doc_path).open("r", encoding = "utf-8") as f:
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

    return lines

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