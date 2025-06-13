import torch
import torch.nn as nn
# from chinesebert.modeling_chinesebert import ChineseBertModel
from transformers.models.bert import BertModel, BertPreTrainedModel


model_name2model_cls = {
    "bert": (BertPreTrainedModel, BertModel)
}


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(
        self,
        hidden_size,
        heads=12,
        head_size=64,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * 2 * head_size, bias=use_bias)

    def get_rotary_positions_embeddings(self, inputs, output_dim):
        position_ids = torch.arange(0, inputs.size(1), dtype=inputs.dtype, device=inputs.device)

        indices = torch.arange(0, output_dim // 2, dtype=inputs.dtype, device=inputs.device)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum("n,d->nd", position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], axis=-1).flatten(1, 2)
        return embeddings[None, :, :]

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense(inputs)
        bs, seqlen = inputs.shape[:2]

        # method 1
        inputs = inputs.reshape(bs, seqlen, self.heads, 2, self.head_size)
        qw, kw = inputs.unbind(axis=-2)

        # method 2
        # inputs = inputs.reshape(bs, seqlen, self.heads, 2 * self.head_size)
        # qw, kw = inputs.chunk(2, axis=-1)

        # original
        # inputs = inputs.chunk(self.heads, axis=-1)
        # inputs = torch.stack(inputs, axis=-2)
        # qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]

        # RoPE编码
        if self.RoPE:
            pos = self.get_rotary_positions_embeddings(inputs, self.head_size)
            cos_pos = torch.repeat_interleave(pos[..., None, 1::2], 2, axis=-1)
            sin_pos = torch.repeat_interleave(pos[..., None, ::2], 2, axis=-1)

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], axis=-1).reshape_as(qw)

            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], axis=-1).reshape_as(kw)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)  # bs*heads*seqlen*seqlen

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] *
                attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * 1e12

        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits = logits - mask * 1e12

        # scale返回
        return logits / self.head_size ** 0.5


def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    多标签分类的交叉熵：https://kexue.fm/archives/7359
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    # print(f'neg: {y_pred_neg.shape}; pos: {y_pred_pos.shape}')
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    # print(f'neg: {neg_loss.shape}; pos: {pos_loss.shape}')

    return (neg_loss + pos_loss).mean()


def ner_loss(y_pred, y_true):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size*ent_type_size, -1)  # (batch_size*ent_type_size, seq_len*seq_len)
    y_pred = y_pred.reshape(batch_size*ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


def text_loss(y_pred, y_true):
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(y_pred.view(-1, 9), y_true.view(-1))
    return loss


class GPNERNet(BertPreTrainedModel):
    def __init__(self, config):
        super(GPNERNet, self).__init__(config)
        self.bert = BertModel(config)
        self.entity_output = GlobalPointer(hidden_size=config.hidden_size, heads=config.num_labels)
        # self.entity_output = GlobalPointer(hidden_size=config.hidden_size, heads=config.num_labels, RoPE=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        context_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0]  # last_hidden_state:(batch_size, seq_len, hidden_size)

        entity_output_logits = self.entity_output(last_hidden_state, attention_mask=attention_mask)
        outputs = (entity_output_logits,)

        if labels is not None:
            # loss = re_loss(entity_output_logits, labels)
            loss = ner_loss(entity_output_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits


class GPArgumentForSpecificTriggerNet(BertPreTrainedModel):
    def __init__(self, config):
        super(GPArgumentForSpecificTriggerNet, self).__init__(config)
        self.bert = BertModel(config)
        self.entity_output = GlobalPointer(hidden_size=config.hidden_size, heads=config.num_labels)
        # self.entity_output = GlobalPointer(hidden_size=config.hidden_size, heads=config.num_labels, RoPE=False)
        self.post_init()

    def get_encoded_text(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)  # [batch_size, seq_len, bert_dim(768)]
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output

    def get_objs_for_specific_sub(self, attention_mask, sub_head_mapping, sub_tail_mapping, encoded_text):
        sub_head = torch.matmul(sub_head_mapping, encoded_text)  # [batch_size, 1, bert_dim]
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)  # [batch_size, 1, bert_dim]
        sub = (sub_head + sub_tail) / 2  # [batch_size, 1, bert_dim]
        # sub = (sub_head + sub_tail) / 2 * 10  # [batch_size, 1, bert_dim]
        encoded_text = encoded_text + sub  # [batch_size, seq_len, bert_dim]
        entity_output_logits = self.entity_output(encoded_text, attention_mask=attention_mask)
        return entity_output_logits

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, sub_span=None):
        sequence_output, pooled_output = self.get_encoded_text(input_ids, attention_mask, token_type_ids)  # last_hidden_state:(batch_size, seq_len, hidden_size)
        if sub_span:
            sub_head, sub_tail = sub_span
            sub_head_mapping = sub_head.unsqueeze(1)  # [batch_size, 1, seq_len]
            sub_tail_mapping = sub_tail.unsqueeze(1)  # [batch_size, 1, seq_len]

            entity_output_logits = self.get_objs_for_specific_sub(attention_mask, sub_head_mapping, sub_tail_mapping, sequence_output)
            outputs = (entity_output_logits,)

        if labels is not None:
            loss = ner_loss(entity_output_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits
