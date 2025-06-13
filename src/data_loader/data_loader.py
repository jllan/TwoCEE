
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


logger = logging.getLogger(__name__)

cur_dir = os.path.dirname(os.path.realpath(__file__))


class GPNERDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        max_length,
        ent2id,
        processor=None,
        doc_stride=None,
        input_file=None,
        text_list=None,
        examples=None,
        data_type='train',
        use_num=None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.data_type = data_type
        if not (text_list or input_file or examples):
            raise ValueError('输入数据错误')

        self.processor = processor
        self.ent2id = ent2id

        if text_list:
            examples = [{'text': text, 'entity_list': []} for text in text_list]
        elif input_file:
            examples = self.processor.read_data(input_file=input_file, data_type=data_type)
            if use_num:
                examples = examples[:use_num]
        self.features = self.convert_examples_to_features(examples)

    def __getitem__(self, index):
        feature = self.features[index]
        return feature

    def __len__(self):
        return len(self.features)

    def get_ent2token_spans(self, offset_mapping, ent_char_start, ent_char_end, add_special_tokens=True):
        """实体列表转为token_spans

        Args:
            text (str): 原始文本
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """
        token2char_span_mapping = offset_mapping

        # print(f'char-position: {ent_char_start}, {ent_char_end}')
        ent_char_end += 1

        ent_span_start, ent_span_end = None, None
        for i, offset in enumerate(token2char_span_mapping[1: -1], 1):
            if offset[0] == ent_char_start:
                ent_span_start = i
            if offset[1] == ent_char_end:
                ent_span_end = i
            if ent_span_start and ent_span_end:
                break

        return ent_span_start, ent_span_end

    def convert_examples_to_features(self, examples):
        features = []
        for example in examples:
            if 'prompt' in example:
                tokenized_example = self.tokenizer(
                    example['prompt'],
                    example['text'],
                    stride=0 if self.doc_stride is None else self.doc_stride,
                    max_length=self.max_length,
                    padding=False,
                    truncation='only_second',
                    return_offsets_mapping=True,
                    return_overflowing_tokens=True,
                    return_token_type_ids=True,
                )
            else:
                tokenized_example = self.tokenizer(
                    example['text'],
                    stride=0 if self.doc_stride is None else self.doc_stride,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_offsets_mapping=True,
                    return_overflowing_tokens=True,
                    return_token_type_ids=True,
                )

            offset_mappings = tokenized_example.pop('offset_mapping')
            if self.doc_stride is None:  # 如果未指定stride，说明无需滑窗，使用第一个窗口的内容即可
                offset_mappings = offset_mappings[:1]
            for i, offsets in enumerate(offset_mappings):
                input_ids = tokenized_example["input_ids"][i]
                attention_mask = tokenized_example["attention_mask"][i]
                token_type_ids = tokenized_example["token_type_ids"][i]
                feature = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

                offsets = [list(x) for x in offsets]  # tuple-->list

                if 'prompt' in example:
                    prompt_end_token_id = offsets[1:-1].index([0, 0])+1  # 找到prompt和text之间的[SEP]的索引
                    bias = offsets[prompt_end_token_id-1][1] + 1
                    # for index in range(bias+1, len(offsets)-1):
                    for index in range(prompt_end_token_id+1, len(offsets)-1):
                        offsets[index][0] += bias
                        offsets[index][1] += bias
                else:
                    bias = 0

                # bias = 0
                # for index in range(1, len(offsets)-1):
                #     mapping = offsets[index]
                #     if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                #         bias = offsets[index - 1][1] + 1  # Includes [SEP] token
                #         print('bias 2..............: ', bias)
                #     if mapping[0] == 0 and mapping[1] == 0:
                #         continue
                #     offsets[index][0] += bias
                #     offsets[index][1] += bias

                example_new = example.copy()
                example_new['offset_mapping'] = offsets

                if self.data_type != 'test':
                    label = []  # 存储当前窗口内的实体的位置
                    entities_new = []  # 存储当前窗口内的实体

                    for i, ent in enumerate(example_new['entity_list']):
                        # ent_char_start = ent['ent_start_inx']
                        ent_char_start = ent['ent_start_inx'] + bias
                        ent_char_end = ent_char_start+len(ent['ent_text'])-1
                        ent_token_start = tokenized_example.char_to_token(ent_char_start)
                        ent_token_end = tokenized_example.char_to_token(ent_char_end)  # sequence_index指定第几个句子
                        # ent_token_start, ent_token_end = self.get_ent2token_spans(offsets, ent_char_start, ent_char_end)  # 如果有prompt，修改了offset_mapping，使用get_ent2token_spans找到char对应的token的位置
                        ent_type_id = self.ent2id[ent['ent_type']]

                        if ent_token_start is None or ent_token_end is None:
                            print(f'cannot find {ent} in {example["text"]}')
                            continue

                        label.append([ent_token_start, ent_token_end, ent_type_id])
                        # print([ent_token_start, ent_token_end, ent_type_id])
                        ent['ent_start_inx'] = ent_char_start
                        entities_new.append(ent)

                    if self.data_type == 'train':  # 对于训练集，如果当前窗口内没有实体，即label为空，则丢弃当前窗口的内容
                        feature['labels'] = label
                        features.append(feature)
                    if self.data_type == 'eval':
                        if self.doc_stride is not None:
                            # 如果指定了stride，说明需要滑窗，那对于验证集来说，每一次只取当前窗口内的实体进行验证即可；
                            # 如果未指定stride，说明无需滑窗，当前窗口的数据就代表了整条数据，验证时就应该使用该数据所有的实体
                            example_new['entity_list'] = entities_new
                        feature['raw_data'] = example_new
                        features.append(feature)

                else:
                    feature['raw_data'] = example_new
                    features.append(feature)
        return features


class GPArgumentDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        max_length,
        ent2id,
        processor=None,
        doc_stride=None,
        input_file=None,
        text_list=None,
        examples=None,
        data_type='train',
        use_num=None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.data_type = data_type
        if not (text_list or input_file or examples):
            raise ValueError('输入数据错误')

        self.processor = processor
        self.ent2id = ent2id

        if text_list:
            examples = [{'text': text, 'entity_list': []} for text in text_list]
        elif input_file:
            examples = self.processor.read_data(input_file=input_file, data_type=data_type)
            if use_num:
                examples = examples[:use_num]
        self.features = self.convert_examples_to_features(examples)

    def __getitem__(self, index):
        feature = self.features[index]
        return feature

    def __len__(self):
        return len(self.features)

    def get_ent2token_spans(self, offset_mapping, ent_char_start, ent_char_end, add_special_tokens=True):
        """实体列表转为token_spans

        Args:
            text (str): 原始文本
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """
        token2char_span_mapping = offset_mapping

        # print(f'char-position: {ent_char_start}, {ent_char_end}')
        ent_char_end += 1

        ent_span_start, ent_span_end = None, None
        for i, offset in enumerate(token2char_span_mapping[1: -1], 1):
            if offset[0] == ent_char_start:
                ent_span_start = i
            if offset[1] == ent_char_end:
                ent_span_end = i
            if ent_span_start and ent_span_end:
                break

        return ent_span_start, ent_span_end

    def convert_examples_to_features(self, examples):
        features = []
        for example in examples:
            if 'prompt' in example:
                tokenized_example = self.tokenizer(
                    example['prompt'],
                    example['text'],
                    stride=0 if self.doc_stride is None else self.doc_stride,
                    max_length=self.max_length,
                    padding=False,
                    truncation='only_second',
                    return_offsets_mapping=True,
                    return_overflowing_tokens=True,
                    return_token_type_ids=True,
                )
            else:
                tokenized_example = self.tokenizer(
                    example['text'],
                    stride=0 if self.doc_stride is None else self.doc_stride,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_offsets_mapping=True,
                    return_overflowing_tokens=True,
                    return_token_type_ids=True,
                )

            offset_mappings = tokenized_example.pop('offset_mapping')
            if self.doc_stride is None:  # 如果未指定stride，说明无需滑窗，使用第一个窗口的内容即可
                offset_mappings = offset_mappings[:1]
            for i, offsets in enumerate(offset_mappings):
                input_ids = tokenized_example["input_ids"][i]
                attention_mask = tokenized_example["attention_mask"][i]
                token_type_ids = tokenized_example["token_type_ids"][i]
                feature = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

                offsets = [list(x) for x in offsets]  # tuple-->list

                if 'prompt' in example:
                    prompt_end_token_id = offsets[1:-1].index([0, 0])+1  # 找到prompt和text之间的[SEP]的索引
                    bias = offsets[prompt_end_token_id-1][1] + 1
                    # for index in range(bias+1, len(offsets)-1):
                    for index in range(prompt_end_token_id+1, len(offsets)-1):
                        offsets[index][0] += bias
                        offsets[index][1] += bias
                else:
                    bias = 0

                example_new = example.copy()
                example_new['offset_mapping'] = offsets

                sub_char_start, sub_char_end = example_new['sub_span']  # [trig_start_inx, trig_end_inx]
                sub_char_end = sub_char_end-1
                sub_token_start = tokenized_example.char_to_token(sub_char_start)
                sub_token_end = tokenized_example.char_to_token(sub_char_end)  # sequence_index指定第几个句子
                feature['sub_span'] = [sub_token_start, sub_token_end]

                if self.data_type != 'test':
                    label = []  # 存储当前窗口内的实体的位置
                    entities_new = []  # 存储当前窗口内的实体

                    for i, ent in enumerate(example_new['entity_list']):
                        # ent_char_start = ent['ent_start_inx']
                        ent_char_start = ent['ent_start_inx'] + bias
                        ent_char_end = ent_char_start+len(ent['ent_text'])-1
                        ent_token_start = tokenized_example.char_to_token(ent_char_start)
                        ent_token_end = tokenized_example.char_to_token(ent_char_end)  # sequence_index指定第几个句子
                        # ent_token_start, ent_token_end = self.get_ent2token_spans(offsets, ent_char_start, ent_char_end)  # 如果有prompt，修改了offset_mapping，使用get_ent2token_spans找到char对应的token的位置
                        ent_type_id = self.ent2id[ent['ent_type']]

                        if ent_token_start is None or ent_token_end is None:
                            print(f'cannot find {ent} in {example["text"]}')
                            continue

                        label.append([ent_token_start, ent_token_end, ent_type_id])
                        # print([ent_token_start, ent_token_end, ent_type_id])
                        ent['ent_start_inx'] = ent_char_start
                        entities_new.append(ent)

                    if self.data_type == 'train':  # 对于训练集，如果当前窗口内没有实体，即label为空，则丢弃当前窗口的内容
                        # feature['labels'] = (label, [sub_token_start, sub_token_end])
                        feature['labels'] = label
                        features.append(feature)
                    if self.data_type == 'eval':
                        if self.doc_stride is not None:
                            # 如果指定了stride，说明需要滑窗，那对于验证集来说，每一次只取当前窗口内的实体进行验证即可；
                            # 如果未指定stride，说明无需滑窗，当前窗口的数据就代表了整条数据，验证时就应该使用该数据所有的实体
                            example_new['entity_list'] = entities_new
                        feature['raw_data'] = example_new
                        features.append(feature)

                else:
                    feature['raw_data'] = example_new
                    features.append(feature)
        return features


@dataclass
class DataCollatorForGPNER:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        ignore_list = ['raw_data']
        new_features = [
            {k: v for k, v in f.items() if k not in ["labels"] + ignore_list}
            for f in features
        ]
        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if labels is None:  # for test
            batch['raw_data'] = [feature["raw_data"] for feature in features]
            return batch

        bs = batch["input_ids"].size(0)
        max_length = batch["input_ids"].size(1)
        # max_ent_num = max([len(lb) for lb in labels])
        # batch_entity_labels = torch.zeros(bs, self.num_labels, max_ent_num, 2, dtype=torch.long)
        batch_entity_labels = torch.zeros(bs, self.num_labels, max_length, max_length, dtype=torch.long)
        for i, lb in enumerate(labels):
            for spidx, (eh, et, p) in enumerate(lb):
                # batch_entity_labels[i, p, spidx, :] = torch.tensor([eh, et])
                # labels[ent2id[label], start, end] = 1
                batch_entity_labels[i, p, eh, et] = 1

        batch["labels"] = batch_entity_labels
        return batch


@dataclass
class DataCollatorForGPArgument:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        sub_spans = (
            [feature["sub_span"] for feature in features]
            if "sub_span" in features[0].keys()
            else None
        )
        ignore_list = ['raw_data']
        new_features = [
            {k: v for k, v in f.items() if k not in ["labels"] + ignore_list}
            for f in features
        ]
        batch = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        bs = batch["input_ids"].size(0)
        max_length = batch["input_ids"].size(1)
        batch_trigger_head, batch_trigger_tail = torch.zeros(bs, max_length, dtype=torch.float), torch.zeros(bs, max_length, dtype=torch.float)
        for i, sub_span in enumerate(sub_spans):
            batch_trigger_head[i, sub_span[0]] = 1
            batch_trigger_tail[i, sub_span[1]] = 1
        batch['sub_span'] = [batch_trigger_head, batch_trigger_tail]

        if labels is None:  # for test
            batch['raw_data'] = [feature["raw_data"] for feature in features]
            return batch

        batch_entity_labels = torch.zeros(bs, self.num_labels, max_length, max_length, dtype=torch.long)
        for i, lb in enumerate(labels):
            for spidx, (eh, et, p) in enumerate(lb):
                batch_entity_labels[i, p, eh, et] = 1
        batch["labels"] = batch_entity_labels

        return batch
