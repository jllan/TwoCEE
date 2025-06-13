
import os
import sys
import json
import math
import logging

from tqdm import tqdm, trange
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertConfig, set_seed, get_scheduler

cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_dir, '../../src'))

from utils.postprocess import decode_ent
from utils.args import parse_args
from data_loader.data_reader import data_processors
from utils.adversarial import FGM, PGD


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

# os.environ["TOKENIZERS_PARALLELISM"] = "true"


def train(args, model, train_dataloader, tokenizer, eval_dataloader=None):
    tb_writer = SummaryWriter(args.output_dir)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # # scheduler
    # cawr_scheduler = {
    #     # CosineAnnealingWarmRestarts
    #     "T_mult": 1,
    #     "rewarm_epoch_num": 2,
    # }
    # step_scheduler = {
    #     # StepLR
    #     "decay_rate": 0.999,
    #     "decay_steps": 100,
    # }
    # scheduler_type = 'CAWR'  # or 'Step'
    # if scheduler_type == "CAWR":
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_dataloader) * cawr_scheduler["rewarm_epoch_num"], cawr_scheduler['T_mult'])
    # elif scheduler_type == "Step":
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_scheduler['decay_steps'], gamma=step_scheduler['decay_rate'])

    no_decay = ["bias", "LayerNorm.weight", "norm"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
    args.num_warmup_steps = (
        math.ceil(args.max_train_steps * args.num_warmup_steps_or_radios)
        if isinstance(args.num_warmup_steps_or_radios, float)
        else args.num_warmup_steps_or_radios
    )
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.per_device_train_batch_size
        * max(1, args.n_gpu)
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)

    if args.adv_type == 'fgm':
        fgm = FGM(model, emb_name='word_embeddings', epsilon=1.0)
    if args.adv_type == 'pgd':
        pgd = PGD(model, emb_name='word_embeddings.', epsilon=1.0, alpha=0.3)

    global_step = 0
    epochs_trained = 0

    tr_loss = 0.0
    logging_loss = 0.0
    best_f1 = 0.0

    # model.zero_grad()
    dev_score_records = []
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()

            for k, v in batch.items():
                if k in ['sub_span']:
                    batch[k] = [vv.to(args.device) for vv in v]
                else:
                    batch[k] = v.to(args.device)

            loss = model(**batch)[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()

            if args.adv_type == 'fgm':
                fgm.attack()  # 对抗训练
                loss_adv = model(**batch)[0]
                # if args.n_gpu > 1:
                loss_adv = loss_adv.mean()
                loss_adv = loss_adv / args.gradient_accumulation_steps
                loss_adv.backward()
                fgm.restore()

            if args.adv_type == 'pgd':
                K = 3
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t==0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K-1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv = model(**batch)[0]
                    loss_adv = loss_adv.mean()
                    loss_adv = loss_adv / args.gradient_accumulation_steps
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参数

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                # model.zero_grad()
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[-1], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    current_tr_loss = (tr_loss - logging_loss) / args.logging_steps
                    logger.info(f"epoch: {epoch}, global_step: {global_step}, lr: {scheduler.get_last_lr()[-1]}, train_loss: {current_tr_loss}")
                    logging_loss = tr_loss

                # if args.local_rank in [-1, 0] and (args.logging_steps > 0 and global_step % args.logging_steps == 0) or global_step == args.max_train_steps:

                    # if global_step < 1000:
                    #     continue

        logger.info(f"{'-' * 20} Evaluate Step {global_step} {'-' * 20}")
        eval_metric, _ = evaluate(args, model, eval_dataloader, need_position=False)
        # care_indicator = ['micro', 'macro'][0]
        # eval_metric = eval_metric[care_indicator]
        # for k, v in eval_metric.items():
        #     logger.info(f"{k} = {v}")
        #     tb_writer.add_scalar(f"eval_{k}", v, global_step)
        logger.info(eval_metric)
        logger.info(f"{'-' * 20} Evaluate End {'-' * 20}")

        eval_metric['global_step'] = global_step

        dev_score_records.append(list(eval_metric['micro'].values())+list(eval_metric['macro'].values())+[global_step, current_tr_loss])
        columns = [f'micro-{i}' for i in eval_metric['micro'].keys()] + [f'macro-{i}' for i in eval_metric['macro'].keys()] + ['global_steps', 'train_loss']
        records_df = pd.DataFrame(dev_score_records, columns=columns)
        records_df.to_csv(os.path.join(args.output_dir, 'eval_records.csv'), index=False)

        current_eval_f1 = eval_metric['micro'].get('f1', 0.0)
        if current_eval_f1 > best_f1:
            # 保存F1最高的模型
            output_eval_file = os.path.join(args.output_dir, "eval_results.json")
            with open(output_eval_file, 'w') as f:
                f.write(json.dumps(eval_metric, ensure_ascii=False, indent=2))

            logger.info(f'当前最高f1为: {current_eval_f1}, 保存当前模型.........')
            best_f1 = current_eval_f1

            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.info("Saving model checkpoint to %s", args.output_dir)
            logger.info("*************************************")

        # manually release the unused cache
        torch.cuda.empty_cache()

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataloader, need_position=True, threshold=0, need_prob=False):
    model.eval()

    true_ents_all, pred_ents_all = [], []
    ids_all, texts_all = [], []
    metric_info = dict()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        batch_samples = batch.pop('raw_data')
        offset_mappings = [sample['offset_mapping'] for sample in batch_samples]
        texts = [sample['text'] for sample in batch_samples]
        if 'prompt' in batch_samples[0]:
            prompts = [sample['prompt'] for sample in batch_samples]
        else:
            # prompts = None
            prompts = [None]*len(texts)
        true_ents = [sample['entity_list'] for sample in batch_samples]
        ids = [sample['id'] for sample in batch_samples]

        for k, v in batch.items():
            if k in ['sub_span']:
                batch[k] = [vv.to(args.device) for vv in v]
            else:
                batch[k] = v.to(args.device)

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs[0]  # entity_outputs, head_outputs, tail_outputs

        pred_ents = decode_ent(args, logits, offset_mappings, texts, prompts, need_position=need_position, threshold=threshold, need_prob=need_prob)  # [((ent_text, ent_char_start), ent_type, ent_prob)]

        if need_position:
            true_ents = [
                [
                    ((ent['ent_text'], ent['ent_start_inx']), ent['ent_type']) for ent in golds
                ] for golds in true_ents
            ]
        else:
            true_ents = [
                [
                    (ent['ent_text'], ent['ent_type']) for ent in golds
                ] for golds in true_ents
            ]

        assert len(pred_ents) == len(true_ents) == len(texts)

        pred_ents_all.extend(pred_ents)
        true_ents_all.extend(true_ents)
        texts_all.extend(texts)
        ids_all.extend(ids)

    eval_results = dict()
    eval_results_difference = list()
    for id_, text, true_ents, pred_ents in zip(ids_all, texts_all, true_ents_all, pred_ents_all):
        true_ents = set(true_ents)
        pred_ents = set(pred_ents)
        # true_ents = {i for i in true_ents if i[0][0] != '[N]'}
        # pred_ents = {i for i in pred_ents if i[0][0] != '[N]'}
        if id_ not in eval_results:
            eval_results[id_] = dict()
            eval_results[id_]['id'] = id_
            eval_results[id_]['text'] = text
            eval_results[id_]['ent_list'] = true_ents
            eval_results[id_]['ent_list_pred'] = pred_ents
        else:
            eval_results[id_]['ent_list'] |= true_ents
            eval_results[id_]['ent_list_pred'] |= pred_ents

    metric_info['micro'] = {'correct_num': 0, 'pred_num': 0, 'true_num': 0}
    metric_info['macro'] = {'correct_num': 0, 'pred_num': 0, 'true_num': 0}
    metric_info['each_label'] = dict()
    for id_, data in eval_results.items():
        golds = data['ent_list']
        preds = data['ent_list_pred']
        print(f'**************** {id_} ****************')
        print(f'true........: {golds}')
        print(f'pred........: {preds}')
        if need_prob:
            corrects = {p[:-1] for p in preds} & golds
        else:
            corrects = preds & golds

        metric_info['micro']['correct_num'] += len(corrects)
        metric_info['micro']['pred_num'] += len(preds)
        metric_info['micro']['true_num'] += len(golds)
        for dt, ents in [('correct', corrects), ('pred', preds), ('true', golds)]:
            for ent in ents:
                et = ent[1]
                if et not in metric_info['each_label']:
                    metric_info['each_label'][et] = {'correct_num': 0, 'pred_num': 0, 'true_num': 0}
                metric_info['each_label'][et][f'{dt}_num'] += 1

        entity_list = [' | '.join(['-'.join([str(k) for k in j]) if isinstance(j, tuple) else j for j in i]) for i in list(golds)]
        entity_list_pred = [' | '.join(['-'.join([str(k) for k in j]) if isinstance(j, tuple) else j for j in i]) for i in list(preds)]
        if need_prob:
            preds = {p[:-1] for p in preds}  # 去除实体的概率值
        new = [' | '.join(['-'.join([str(k) for k in j]) if isinstance(j, tuple) else j for j in i]) for i in list(preds-golds)]
        lack = [' | '.join(['-'.join([str(k) for k in j]) if isinstance(j, tuple) else j for j in i]) for i in list(golds-preds)]
        eval_results_difference.append(
            {
                "id": data['id'],
                "text": data['text'],
                "entity_list": entity_list,
                "entity_list_pred": entity_list_pred,
                "new": new,
                "lack": lack,
                'is_corrected': True if preds == golds else False
            }
        )

    p_micro = metric_info['micro']['correct_num'] / (metric_info['micro']['pred_num'] + 1e-10)
    r_micro = metric_info['micro']['correct_num'] / (metric_info['micro']['true_num'] + 1e-10)
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-10)
    metric_info['micro'].update({'precision': p_micro, 'recall': r_micro, 'f1': f1_micro})

    p_macro, r_macro, f1_macro = 0., 0., 0.
    for p in metric_info['each_label']:
        tn = metric_info['each_label'][p]["true_num"]
        pn = metric_info['each_label'][p]["pred_num"]
        cn = metric_info['each_label'][p]["correct_num"]
        precision = cn / (pn + 1e-10)
        recall = cn / (tn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        metric_info['each_label'][p].update({'precision': precision, 'recall': recall, 'f1': f1})
        p_macro += precision
        r_macro += recall
        f1_macro += f1
    metric_info['macro'] = {'precision': p_macro/len(metric_info['each_label']), 'recall': r_macro/len(metric_info['each_label']), 'f1': f1_macro/len(metric_info['each_label'])}

    print("******************************************")
    print(metric_info)
    print("******************************************")

    return metric_info, eval_results_difference


def predict(args, model, test_dataloader, need_position=True):
    model.eval()

    pred_ents_all = []
    ids_all, texts_all = [], []
    for batch in tqdm(test_dataloader, desc="Predicting"):

        batch_samples = batch.pop('raw_data')
        offset_mappings = [sample['offset_mapping'] for sample in batch_samples]
        texts = [sample['text'] for sample in batch_samples]
        ids = [sample['id'] for sample in batch_samples]
        if 'prompt' in batch_samples[0]:
            prompts = [sample['prompt'] for sample in batch_samples]
        else:
            # prompts = None
            prompts = [None]*len(texts)

        for k, v in batch.items():
            batch[k] = v.to(args.device)

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs[0]  # entity_outputs, head_outputs, tail_outputs

        pred_ents = decode_ent(args, logits, offset_mappings, texts, prompts, need_position=need_position)

        assert len(pred_ents) == len(texts)

        pred_ents_all.extend(pred_ents)
        texts_all.extend(texts)
        ids_all.extend(ids)

    predict_results = dict()
    for id_, text, pred_ents in zip(ids_all, texts_all, pred_ents_all):
        pred_ents = set(pred_ents)
        if id_ not in predict_results:
            predict_results[id_] = dict()
            predict_results[id_]['id'] = id_
            predict_results[id_]['text'] = text
            predict_results[id_]['ent_list_pred'] = pred_ents
        else:
            predict_results[id_]['ent_list_pred'] |= pred_ents

    for data in predict_results.values():
        preds = data['ent_list_pred']
        preds = sorted(preds, key=lambda x: x[0][1], reverse=False)
        data['ent_list_pred'] = preds

    return predict_results


def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    stage = ['trigger', 'argument', 'argument_embedding'][2]

    processor = data_processors[args.task_name](stage=stage, max_length=args.max_length)
    ent2id, id2ent = processor.get_schema(args.schema_file)
    print(ent2id)
    print(id2ent)
    args.ent2id = ent2id
    args.id2ent = id2ent
    args.num_labels = len(ent2id)

    if args.do_train:
        if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        if args.do_train and (not os.path.exists(args.output_dir)):
            os.mkdir(args.output_dir)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    os.path.join(args.output_dir, "run.log"),
                    mode="w",
                    encoding="utf-8",
                )
            ],
        )

        tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, do_lower_case=True)
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
        if stage == 'argument_embedding':
            from models.model import GPArgumentForSpecificTriggerNet as GPNERNet
            from data_loader.data_loader import GPArgumentDataset as GPNERDataset
            from data_loader.data_loader import DataCollatorForGPArgument as DataCollatorForGPNER
        else:
            from models.model import GPNERNet
            from data_loader.data_loader import GPNERDataset, DataCollatorForGPNER

        model = GPNERNet.from_pretrained(args.model_name_or_path, config=config)
        model.to(args.device)

        data_collator = DataCollatorForGPNER(
            tokenizer,
            pad_to_multiple_of=(8 if args.fp16 else None),
            num_labels=args.num_labels,
        )

        train_dataset = (
            GPNERDataset(
                processor=processor,
                tokenizer=tokenizer,
                ent2id=args.ent2id,
                input_file=args.train_data_file,
                max_length=args.max_length,
                data_type='train',
                # use_num=1587//10,
            )
            if args.do_train
            else None
        )
        # batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size * max(1, args.n_gpu),
            shuffle=True,
            pin_memory=True,
            collate_fn=data_collator
        )
        eval_dataset = (
            GPNERDataset(
                processor=processor,
                tokenizer=tokenizer,
                ent2id=args.ent2id,
                input_file=args.eval_data_file,
                max_length=args.max_length,
                data_type='eval',
                # use_num=1,
            )
            if args.do_eval
            else None
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size * max(1, args.n_gpu),
            shuffle=False,
            pin_memory=True,
            collate_fn=data_collator
        )

        train(args, model, train_dataloader, tokenizer, eval_dataloader)

    args.do_eval = False

    if args.do_eval:
        tokenizer = BertTokenizerFast.from_pretrained(args.output_dir, do_lower_case=True)
        model = GPNERNet.from_pretrained(args.output_dir)
        model.to(args.device)

        data_collator = DataCollatorForGPNER(
            tokenizer,
            pad_to_multiple_of=(8 if args.fp16 else None),
            num_labels=args.num_labels,
        )

        eval_dataset = GPNERDataset(
            processor=processor,
            tokenizer=tokenizer,
            ent2id=args.ent2id,
            input_file=args.eval_data_file,
            max_length=args.max_length,
            data_type='eval',
            # use_num=10,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size * max(1, args.n_gpu),
            shuffle=False,
            pin_memory=True,
            collate_fn=data_collator
        )

        thres = 0.0
        eval_metric, eval_results = evaluate(args, model, eval_dataloader, need_position=False, threshold=thres, need_prob=False)

    args.do_predict = False
    if args.do_predict:
        tokenizer = BertTokenizerFast.from_pretrained(args.output_dir, do_lower_case=True)
        model = GPNERNet.from_pretrained(args.output_dir)
        model.to(args.device)

        data_collator = DataCollatorForGPNER(
            tokenizer,
            pad_to_multiple_of=(8 if args.fp16 else None),
            num_labels=args.num_labels,
        )
        test_dataset = GPNERDataset(
            processor=processor,
            tokenizer=tokenizer,
            ent2id=args.ent2id,
            input_file=args.test_data_file,
            max_length=args.max_length,
            data_type='test'
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.per_device_eval_batch_size * max(1, args.n_gpu),
            shuffle=False,
            pin_memory=True,
            collate_fn=data_collator
        )

        predict_results = predict(args, model, test_dataloader, tokenizer)
        # print(predict_results)
        output_predict_result_file = os.path.join(args.output_dir, "test_pred_results.json")
        predict_results_new = []
        for data in predict_results.values():
            es = []
            data.pop('text')
            pred_ents = data.pop('ent_list_pred')
            if pred_ents:
                for pe in pred_ents:
                    (ent, ent_start_inx), ent_type = pe
                    es.append({
                        'event_type': ent_type,
                        'trigger': {'text': ent, 'offset': [ent_start_inx, ent_start_inx+len(ent)]}
                    })
            data['event_list'] = es
            predict_results_new.append(data)
        with open(output_predict_result_file, 'w') as f:
            f.write(json.dumps(predict_results_new, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
