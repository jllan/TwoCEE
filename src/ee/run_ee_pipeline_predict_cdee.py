import os
import sys
import json
import math
import logging
from collections import defaultdict

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, set_seed

cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_dir, "../../src"))

from utils.postprocess import decode_ent
from models.model import GPNERNet
from models.model import GPArgumentForSpecificTriggerNet
from data_loader.data_loader import GPNERDataset, DataCollatorForGPNER
from data_loader.data_loader import GPArgumentDataset, DataCollatorForGPArgument
from utils.args import parse_args
from data_loader.data_reader import data_processors


logger = logging.getLogger(__name__)


def evaluate(
    args, model, eval_dataloader, need_position=True, threshold=0, need_prob=False
):
    model.eval()

    true_ents_all, pred_ents_all = [], []
    ids_all, texts_all = [], []
    metric_info = dict()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch_samples = batch.pop("raw_data")
        offset_mappings = [sample["offset_mapping"] for sample in batch_samples]
        texts = [sample["text"] for sample in batch_samples]
        if "prompt" in batch_samples[0]:
            prompts = [sample["prompt"] for sample in batch_samples]
        else:
            # prompts = None
            prompts = [None] * len(texts)
        true_ents = [sample["entity_list"] for sample in batch_samples]
        ids = [sample["id"] for sample in batch_samples]

        for k, v in batch.items():
            batch[k] = v.to(args.device)

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs[0]  # entity_outputs, head_outputs, tail_outputs

        pred_ents = decode_ent(
            args,
            logits,
            offset_mappings,
            texts,
            prompts,
            need_position=need_position,
            threshold=threshold,
            need_prob=need_prob,
        )  # [((ent_text, ent_char_start), ent_type, ent_prob)]

        if need_position:
            true_ents = [
                [
                    ((ent["ent_text"], ent["ent_start_inx"]), ent["ent_type"])
                    for ent in golds
                ]
                for golds in true_ents
            ]
        else:
            true_ents = [
                [(ent["ent_text"], ent["ent_type"]) for ent in golds]
                for golds in true_ents
            ]

        assert len(pred_ents) == len(true_ents) == len(texts)

        pred_ents_all.extend(pred_ents)
        true_ents_all.extend(true_ents)
        texts_all.extend(texts)
        ids_all.extend(ids)

    eval_results = dict()
    eval_results_difference = list()
    for id_, text, true_ents, pred_ents in zip(
        ids_all, texts_all, true_ents_all, pred_ents_all
    ):
        true_ents = set(true_ents)
        pred_ents = set(pred_ents)
        if id_ not in eval_results:
            eval_results[id_] = dict()
            eval_results[id_]["id"] = id_
            eval_results[id_]["text"] = text
            eval_results[id_]["ent_list"] = true_ents
            eval_results[id_]["ent_list_pred"] = pred_ents
        else:
            eval_results[id_]["ent_list"] |= true_ents
            eval_results[id_]["ent_list_pred"] |= pred_ents

    metric_info["micro"] = {"correct_num": 0, "pred_num": 0, "true_num": 0}
    metric_info["macro"] = {"correct_num": 0, "pred_num": 0, "true_num": 0}
    metric_info["each_label"] = dict()
    for id_, data in eval_results.items():
        golds = data["ent_list"]
        preds = data["ent_list_pred"]
        print(f"**************** {id_} ****************")
        print(f"true........: {golds}")
        print(f"pred........: {preds}")
        if need_prob:
            corrects = {p[:-1] for p in preds} & golds
        else:
            corrects = preds & golds

        metric_info["micro"]["correct_num"] += len(corrects)
        metric_info["micro"]["pred_num"] += len(preds)
        metric_info["micro"]["true_num"] += len(golds)
        for dt, ents in [("correct", corrects), ("pred", preds), ("true", golds)]:
            for ent in ents:
                et = ent[1]
                if et not in metric_info["each_label"]:
                    metric_info["each_label"][et] = {
                        "correct_num": 0,
                        "pred_num": 0,
                        "true_num": 0,
                    }
                metric_info["each_label"][et][f"{dt}_num"] += 1

        entity_list = [
            " | ".join(
                [
                    "-".join([str(k) for k in j]) if isinstance(j, tuple) else j
                    for j in i
                ]
            )
            for i in list(golds)
        ]
        entity_list_pred = [
            " | ".join(
                [
                    "-".join([str(k) for k in j]) if isinstance(j, tuple) else j
                    for j in i
                ]
            )
            for i in list(preds)
        ]
        if need_prob:
            preds = {p[:-1] for p in preds}  # 去除实体的概率值
        new = [
            " | ".join(
                [
                    "-".join([str(k) for k in j]) if isinstance(j, tuple) else j
                    for j in i
                ]
            )
            for i in list(preds - golds)
        ]
        lack = [
            " | ".join(
                [
                    "-".join([str(k) for k in j]) if isinstance(j, tuple) else j
                    for j in i
                ]
            )
            for i in list(golds - preds)
        ]
        eval_results_difference.append(
            {
                "id": data["id"],
                "text": data["text"],
                "entity_list": entity_list,
                "entity_list_pred": entity_list_pred,
                "new": new,
                "lack": lack,
                "is_corrected": True if preds == golds else False,
            }
        )

    p_micro = metric_info["micro"]["correct_num"] / (
        metric_info["micro"]["pred_num"] + 1e-10
    )
    r_micro = metric_info["micro"]["correct_num"] / (
        metric_info["micro"]["true_num"] + 1e-10
    )
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-10)
    metric_info["micro"].update(
        {"precision": p_micro, "recall": r_micro, "f1": f1_micro}
    )

    p_macro, r_macro, f1_macro = 0.0, 0.0, 0.0
    for p in metric_info["each_label"]:
        tn = metric_info["each_label"][p]["true_num"]
        pn = metric_info["each_label"][p]["pred_num"]
        cn = metric_info["each_label"][p]["correct_num"]
        precision = cn / (pn + 1e-10)
        recall = cn / (tn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        metric_info["each_label"][p].update(
            {"precision": precision, "recall": recall, "f1": f1}
        )
        p_macro += precision
        r_macro += recall
        f1_macro += f1
    metric_info["macro"] = {
        "precision": p_macro / len(metric_info["each_label"]),
        "recall": r_macro / len(metric_info["each_label"]),
        "f1": f1_macro / len(metric_info["each_label"]),
    }

    print("******************************************")
    print(metric_info)
    print("******************************************")

    return metric_info, eval_results_difference


def predict(args, model, examples, tokenizer, threshold=0, need_position=True):
    data_collator = DataCollatorForGPNER(
        tokenizer,
        pad_to_multiple_of=(8 if args.fp16 else None),
        num_labels=args.num_labels,
    )

    test_dataset = GPNERDataset(
        tokenizer=tokenizer,
        ent2id=args.ent2id,
        examples=examples,
        max_length=args.max_length,
        data_type="test",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=data_collator,
    )

    model.eval()

    pred_ents_all = []
    for batch in test_dataloader:
        batch_samples = batch.pop("raw_data")
        offset_mappings = [sample["offset_mapping"] for sample in batch_samples]
        texts = [sample["text"] for sample in batch_samples]
        if "prompt" in batch_samples[0]:
            prompts = [sample["prompt"] for sample in batch_samples]
        else:
            # prompts = None
            prompts = [None] * len(texts)

        for k, v in batch.items():
            batch[k] = v.to(args.device)

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs[0]  # entity_outputs, head_outputs, tail_outputs

        pred_ents = decode_ent(
            args,
            logits,
            offset_mappings,
            texts,
            prompts,
            need_position=need_position,
            threshold=threshold,
        )

        assert len(pred_ents) == len(texts)

        pred_ents_all.extend(pred_ents)

    return pred_ents_all


def predict_argument_embedding(args, model, examples, tokenizer, threshold=0, need_position=True):
    data_collator = DataCollatorForGPArgument(
        tokenizer,
        pad_to_multiple_of=(8 if args.fp16 else None),
        num_labels=args.num_labels,
    )

    test_dataset = GPArgumentDataset(
        tokenizer=tokenizer,
        ent2id=args.ent2id,
        examples=examples,
        max_length=args.max_length,
        data_type="test",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=data_collator,
    )

    model.eval()

    pred_ents_all = []
    for batch in test_dataloader:
        batch_samples = batch.pop("raw_data")
        offset_mappings = [sample["offset_mapping"] for sample in batch_samples]
        texts = [sample["text"] for sample in batch_samples]
        if "prompt" in batch_samples[0]:
            prompts = [sample["prompt"] for sample in batch_samples]
        else:
            # prompts = None
            prompts = [None] * len(texts)

        for k, v in batch.items():
            if k in ['sub_span']:
                batch[k] = [vv.to(args.device) for vv in v]
            else:
                batch[k] = v.to(args.device)

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs[0]  # entity_outputs, head_outputs, tail_outputs

        pred_ents = decode_ent(
            args,
            logits,
            offset_mappings,
            texts,
            prompts,
            need_position=need_position,
            threshold=threshold,
        )

        assert len(pred_ents) == len(texts)

        pred_ents_all.extend(pred_ents)

    return pred_ents_all


def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    trigger_enhance_method = ['char', 'embedding'][0]
    only_comparison = False
    args.do_predict = True
    need_comparison = True
    if need_comparison or only_comparison:
        args.test_data_file = args.eval_data_file
        data_type = "eval"
    else:
        # args.test_data_file = args.eval_data_file
        data_type = "test"

    processor = data_processors[args.task_name](stage="trigger")
    trig2id, id2trig = processor.get_schema(args.schema_file)
    processor = data_processors[args.task_name](stage="argument")
    argu2id, id2argu = processor.get_schema(args.schema_file)
    print(trig2id)
    print(argu2id)

    args.trigger_output_dir = "your-trigger-model-dir"
    args.argument_output_dir = "your-argument-model-dir"

    tokenizer = BertTokenizerFast.from_pretrained(
        args.trigger_output_dir, do_lower_case=True
    )
    trigger_model = GPNERNet.from_pretrained(args.trigger_output_dir)
    trigger_model.to(args.device)
    if trigger_enhance_method == 'char':
        argument_model = GPNERNet.from_pretrained(args.argument_output_dir)
    if trigger_enhance_method == 'embedding':
        argument_model = GPArgumentForSpecificTriggerNet.from_pretrained(args.argument_output_dir)
    argument_model.to(args.device)

    args.output_dir = args.trigger_output_dir

    output_predict_result_file = os.path.join(args.output_dir, f"{data_type}_results.json",)

    if only_comparison:
        eval_results_difference = do_compare(args.test_data_file, output_predict_result_file)
        output_eval_result_file = os.path.join(
            args.output_dir,
            f"{data_type}_difference_results.json",
        )
        with open(output_eval_result_file, "w") as f:
            f.write(json.dumps(eval_results_difference, ensure_ascii=False, indent=2))

    if args.do_predict:
        lines = json.load(open(args.test_data_file, "r"))
        pred_results = []

        for line in tqdm(lines):
            id_ = line["id"]

            events = []

            args.ent2id = trig2id
            args.id2ent = id2trig
            args.num_labels = len(trig2id)
            args.max_length = 300
            examples = [
                {
                    "id": f"{id_}-trigger",
                    "text": line["text"][: args.max_length],
                    "entity_list": [],
                }
            ]

            pred_triggers = predict(args, trigger_model, examples, tokenizer, threshold=trig_thres)[0]
            # print(f'true-es: {line["event"]}')
            # print(f"pred-trigger: {pred_triggers}")

            if pred_triggers:
                args.ent2id = argu2id
                args.id2ent = id2argu
                args.num_labels = len(argu2id)
                args.max_length = 300

                for trigger_info in pred_triggers:
                    (trigger, trigger_char_start), et = trigger_info
                    if trigger == "[N]":
                        continue
                    arguments = []
                    trigger_char_end = trigger_char_start + len(trigger)

                    arguments = {
                        "core_name": trigger,
                        "tendency": et if et != "未知" else "",
                        "character": [],
                        "anatomy_list": [],
                    }

                    text = line["text"]
                    if trigger_enhance_method == 'char':
                        text = (
                            text[:trigger_char_start]
                            + "<T>"
                            + text[trigger_char_start:trigger_char_end]
                            + "</T>"
                            + text[trigger_char_end:]
                        )
                        length_increased = len("<A>") + len("</A>")
                        examples = [{"id": f"{id_}-argument", "text": text, "entity_list": []}]
                        pred_argus = predict(args, argument_model, examples, tokenizer, threshold=argu_thres)[0]  # batch_size=1
                    if trigger_enhance_method == 'embedding':
                        examples = [{"id": f"{id_}-argument", "text": text, "entity_list": [], 'sub_span': [trigger_char_start, trigger_char_end]}]
                        pred_argus = predict_argument_embedding(args, argument_model, examples, tokenizer, threshold=argu_thres)[0]  # batch_size=1

                    for argu_info in pred_argus:
                        (argu, argu_char_start), role = argu_info
                        if argu == "[N]":
                            continue
                        if trigger_enhance_method == 'char':
                            if argu.endswith("<T"):
                                argu = argu[:-2] + trigger
                            if argu_char_start >= trigger_char_end:
                                argu_char_start -= length_increased

                        arguments[role].append(argu)
                    if arguments:
                        events.append(arguments)

            # print(events)

            # 验证集对比
            # "Deploy | 驻扎-83_85 | Subject-朝鲜-18_20, Location-祖国统一战争出发阵地-86_96, Militaryforce-人民军官兵-97_102",

            pred_results.append({"id": id_, "text": line["text"], "event": events})

        def entity_replace(data):
            for line in data:
                es = line['event']
                text = line['text']
                for e in es:
                    trig = e['core_name']
                    if trig == 'ca':
                        e['core_name'] = '癌'
                    if trig == '痛':
                        e['core_name'] = '疼痛'
                    if trig == '痰':
                        e['core_name'] = '咳痰'
                        print(e)
                    if trig == '寒战':
                        e['core_name'] = '寒颤'
                    if trig == '黄':
                        e['core_name'] = '黄染'
                        # print(e)
                    if trig == '体重' and ('改变' in text or '变化' in text):
                        e['core_name'] = '体重改变'
                    if trig == '体重' and '下降' in text:
                        e['core_name'] = '体重下降'
                    if trig == '体重' and '减轻' in text:
                        e['core_name'] = '体重减轻'
                    if trig == '热':
                        e['core_name'] = '发热'
                        print(e)
                    if trig in ['黑蒙', '黑曚']:
                        e['core_name'] = '黑朦'
                        print(e)
            return data

        # pred_results = entity_replace(pred_results)
        with open(output_predict_result_file, "w") as f:
            f.write(json.dumps(pred_results, ensure_ascii=False, indent=2))

        if need_comparison:
            eval_results_difference = do_compare(args.test_data_file, output_predict_result_file)
            output_eval_result_file = os.path.join(
                args.output_dir,
                f"{data_type}_difference_results_trig-{trig_tag}-{trig_thres}_argu-{argu_tag}-{argu_thres}.json",
            )
            with open(output_eval_result_file, "w") as f:
                f.write(json.dumps(eval_results_difference, ensure_ascii=False, indent=2))


def do_compare(origin_data_file, result_data_file):
    true_data = json.load(open(origin_data_file, "r"))
    pred_data = json.load(open(result_data_file, "r"))
    true_num, pred_num, correct_num = 0, 0, 0
    true_num_trigs, pred_num_trigs, correct_num_trigs = 0, 0, 0

    eval_results_difference = []
    for td, pd in zip(true_data, pred_data):
        id_ = td["id"]
        tes = td["event"]
        pes = pd["event"]
        tes_new = []
        pes_new = []
        for e in tes:
            tes_new.append(
                f"{e['core_name']} | {e['tendency']} | {'-'.join(sorted(e['character']))} | {'-'.join(sorted(e['anatomy_list']))}"
            )

        for e in pes:
            pes_new.append(
                f"{e['core_name']} | {e['tendency']} | {'-'.join(sorted(e['character']))} | {'-'.join(sorted(e['anatomy_list']))}"
            )

        # print(tes_new)
        # print(pes_new)
        true_trigs = [i.split("|")[0].strip() for i in tes_new]
        pred_trigs = [i.split("|")[0].strip() for i in pes_new]
        true_num_trigs += len(true_trigs)
        pred_num_trigs += len(pred_trigs)
        correct_num_trigs += len(set(true_trigs) & set(pred_trigs))

        true_num += len(tes_new)
        pred_num += len(pes_new)
        correct_num += len(set(tes_new) & set(pes_new))

        tes_new = sorted(tes_new)
        pes_new = sorted(pes_new)
        eval_results_difference.append(
            {
                "id": id_,
                "event_list": tes_new,
                "event_list_pred": pes_new,
                "new": list(set(pes_new) - set(tes_new)),
                "lack": list(set(tes_new) - set(pes_new)),
                "is_corrected": True if tes_new == pes_new else False,
            }
        )

    p_micro = correct_num_trigs / (pred_num_trigs + 1e-10)
    r_micro = correct_num_trigs / (true_num_trigs + 1e-10)
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-10)
    print(f"true-num-trigs: {true_num_trigs}, pred-num-trigs: {pred_num_trigs}, correct-num-trigs: {correct_num_trigs}")
    print(f"p: {p_micro}, r: {r_micro}, f1: {f1_micro}")

    p_micro = correct_num / (pred_num + 1e-10)
    r_micro = correct_num / (true_num + 1e-10)
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-10)
    print(f"true-num: {true_num}, pred-num: {pred_num}, correct-num: {correct_num}")
    print(f"p: {p_micro}, r: {r_micro}, f1: {f1_micro}")

    return eval_results_difference


if __name__ == "__main__":
    main()
