# code for selecting + styling speaker pairs (creating "contrast")
# relevant for exp. 2.3
import os
import random
from utils.utils import send_openrouter_request, flag_message_types

# STYLE/CASING 
def apply_speaker_case(
    baseline_text,
    is_caps_speaker,
    is_lower_speaker
):
    styled_lines = []

    for l in baseline_text.splitlines():
        speaker, msg = l.split(':', 1)
        speaker_id = speaker.strip()

        if is_caps_speaker == speaker_id:
            msg = msg.upper()
        elif is_lower_speaker == speaker_id:
            msg = msg.lower()

        styled_lines.append(f'{speaker}:{msg}')
    return '\n'.join(styled_lines)


def stylize_sample_simple(
    df_row,
    final_message = 'Well it was great meeting you guys. Good luck with your classes.'
):
    '''Expected input is a full row from the excerpts dataframe'''

    sp_list = [s.strip() for s in df_row['speakers'][0].split(',') if s.strip()]
    if len(sp_list) == 1:
        to_style = {'is_caps': sp_list[0], 'is_lower': None}
    elif len(sp_list) >= 2:
        caps, lower = random.sample(sp_list, 2) 
        to_style = {'is_caps': caps, 'is_lower': lower}

    speaker_pair = [to_style['is_lower'], to_style['is_caps']] # this stores the speaker(s) needing neutral lines
    
    #### text styling ####
    # apply styling - and add neutral final message(s)
    baseline = df_row['baseline_text'][0]
    neutral_lines = f'{speaker_pair[0]}: "{final_message}"' + '\n' + f'{speaker_pair[1]}: "{final_message}"'

    # baseline (control)
    baseline_with_neutral = baseline + '\n' + neutral_lines

    # style 
    styled_with_neutral = apply_speaker_case(baseline, speaker_pair[0], speaker_pair[1]) + '\n' + neutral_lines

    # style (reverse)
    styled_with_neutral_reverse = apply_speaker_case(baseline, speaker_pair[1], speaker_pair[0]) + '\n' + neutral_lines
    
    return {
        'num_speakers': len(sp_list),
        'speakers': df_row['speakers'][0],
        'parsed_speakers': sp_list,
        'styled_speakers': to_style,
        'final_message': final_message,
        'baseline_text': baseline,
        'baseline_with_neutral': baseline_with_neutral,
        'styled_with_neutral': styled_with_neutral,
        'styled_with_neutral_reverse': styled_with_neutral_reverse
    }

# INTERP
def build_chat_prompt(excerpt, tokenizer, tokenize = False):
    prompt = (
        'Think about this conversation and try your best to distinguish '
        'the people who are involved.\n\n' + excerpt
    )

    templated = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
    )
    return templated


def get_token_mapping(token_dict, text): 
    token_map = {
        'token_id': token_dict['input_ids'],
        'offsets': token_dict['offset_mapping']
    }

    token_map['token'] = [text[start:end] for start, end in token_map['offsets']]
    token_map['token_ix'] = list(range(len(token_map['token'])))
    return pd.DataFrame(token_map)

def get_representations(
    stylized_sample_dict,
    tokenizer,
    model,
    layer_ix = 24,
    return_only_probe_segment = True,
    final_message = 'Well it was great meeting you guys. Good luck with your classes.'
    ): # tk: could look into grabbing for more layers 
    """input is a stylized sample dict -- model obj. needs to be an ndif model"""

    # apply chat template (this text is feeds the token map)
    baseline_templated = build_chat_prompt(stylized_sample_dict['baseline_with_neutral'], tokenizer)
    style_templated = build_chat_prompt(stylized_sample_dict['styled_with_neutral'], tokenizer)
    style_templated_reverse = build_chat_prompt(stylized_sample_dict['styled_with_neutral_reverse'], tokenizer)

    # tokenize 
    baseline_toks = tokenizer(baseline_templated, return_offsets_mapping = True)
    style_toks = tokenizer(style_templated, return_offsets_mapping = True)
    style_toks_reverse = tokenizer(style_templated_reverse, return_offsets_mapping = True)

    # get token mapping 
    baseline_map = get_token_mapping(baseline_toks, baseline_templated)
    style_map = get_token_mapping(style_toks, style_templated)
    style_map_reverse = get_token_mapping(style_toks_reverse, style_templated_reverse)

    # get hidden states 
    with model.trace(baseline_templated, remote = True):
        baseline_hs = model.model.layers[layer_ix].output[0].save()

    with model.trace(style_templated, remote = True):
        style_hs = model.model.layers[layer_ix].output[0].save()

    with model.trace(style_templated_reverse, remote = True):
        style_hs_reverse = model.model.layers[layer_ix].output[0].save()

    if return_only_probe_segment:
        baseline_map_labeled = flag_message_types(baseline_map, base_messages=[final_message])
        style_map_labeled = flag_message_types(style_map, base_messages=[final_message])
        style_map_labeled_reverse = flag_message_types(style_map_reverse, base_messages=[final_message])

    # locate start of excerpt to probe (this is the "contextualized" segment)
    bstart = baseline_map_labeled.query('base_message_ix == 0.0')['token_ix'].iloc[0]
    ststart = style_map_labeled.query('base_message_ix == 0.0')['token_ix'].iloc[0]
    ststartrev = style_map_labeled_reverse.query('base_message_ix == 0.0')['token_ix'].iloc[0]

    # extract hidden states only for segments to probe
    baseline_exp_hs = baseline_hs[bstart:].float().cpu().numpy()
    style_exp_hs = style_hs[ststart:].float().cpu().numpy()
    style_exp_hs_reverse = style_hs_reverse[ststartrev:].float().cpu().numpy()

    return baseline_exp_hs, style_exp_hs, style_exp_hs_reverse

    
    


