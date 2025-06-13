import numpy as np


def decode_ent(args, batch_outputs, offset_mappings, texts, prompts, threshold=0, need_position=False, need_prob=False):
    batch_results = []
    # if not prompts:
    #     prompts = [None]*len(texts)
    for entity_output, offset_mapping, text, prompt in zip(
        batch_outputs.cpu().numpy(),
        offset_mappings,
        texts,
        prompts
    ):
        ents = set()
        if not prompt:
            chars = list(text)
        else:
            chars = list(prompt) + ['[SEP]'] + list(text)
        # print(f'entity_output: {entity_output}, {entity_output.shape}')
        for ent_type_id, ent_token_start, ent_token_end in zip(*np.where(entity_output > threshold)):
            ent_type = args.id2ent[ent_type_id]
            ent_char_start = offset_mapping[ent_token_start][0]
            ent_char_end = offset_mapping[ent_token_end][1]
            # ent_text = text[ent_char_start: ent_char_end]
            ent_text = chars[ent_char_start: ent_char_end]
            ent_text = ''.join(ent_text)

            ent_prob = entity_output[ent_type_id][ent_token_start][ent_token_end]
            # if ent_prob < 0.5:
            #     continue

            if not need_position:
                ent = (ent_text, ent_type)
            else:
                ent = ((ent_text, ent_char_start), ent_type)
                # ent = ((ent_text, ent_char_start-len(prompt)-1), ent_type)
            if need_prob:
                ent_prob = str(round(ent_prob, 3))
                ent += (ent_prob,)
            ents.add(ent)

        batch_results.append(list(ents))
    return batch_results


def output_format_cdee(predict_results):
    """
    Output:
    {
        "id": 1636,
        "text": "胸部ct平扫+增强，全腹部ct平扫：1.结肠ca术后改变，肝右叶后段段结节，转移瘤可能。",
        "event": [
            {
                "core_name": "瘤",
                "tendency": "不确定",
                "character": ["转移"],
                "anatomy_list": ["肝右叶后段"]
            }
        ]
    }
    """
    results = []
    for id_, res in predict_results.items():
        pred_spos = res.pop('spo_list_pred')
        events = dict()
        for (s, p, o) in pred_spos:
            # print(s, p, o)
            # sub_text, sub_start = s
            # obj_text, obj_start = o
            # key = f'{sub_text}-{sub_start}'
            sub_text = s
            obj_text = o
            key = sub_text
            if key not in events:
                events[key] = {
                    "core_name": sub_text,
                    "tendency": "",
                    "character": [],
                    "anatomy_list": []
                }
            if obj_text != '未知':
                if p == 'tendency':
                    events[key][p] = obj_text
                if p in ['character', 'anatomy_list']:
                    events[key][p].append(obj_text)
            print(events)

            # if p == 'tendency':
            #     if obj_text != '未知':
            #         events[key][p] = obj_text
            # else:
            #     events[key][p].append(obj_text)
        res['event'] = list(events.values())
        res['id'] = int(res['id'])
        results.append({'id': int(res['id']), 'event': list(events.values())})

    return results
