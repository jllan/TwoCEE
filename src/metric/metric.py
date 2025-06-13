from collections import defaultdict
import json

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_evaluate_fpr(self, results_pred, results_true):

        ent_metric_1 = self.get_evaluate_ent_fpr(results_pred, results_true, filter_empty=False)
        ent_metric_2 = self.get_evaluate_ent_fpr(results_pred, results_true, filter_empty=True)
        et_metric_1 = self.get_evaluate_et_fpr(results_pred, results_true, filter_empty=False)
        et_metric_2 = self.get_evaluate_et_fpr(results_pred, results_true, filter_empty=True)
        event_metric_1 = self.get_evaluate_event_fpr(results_pred, results_true, filter_empty=False)
        event_metric_2 = self.get_evaluate_event_fpr(results_pred, results_true, filter_empty=True)

        metrics = {
            'event': {
                'each_label': event_metric_1['each_label'],
                'micro_fpr': event_metric_1['micro_fpr'],
                'macro_fpr': event_metric_1['macro_fpr'],
                'micro_fpr_no_empty': event_metric_2['micro_fpr'],
                'macro_fpr_no_empty': event_metric_2['macro_fpr']
            },
            'ent': {
                'micro_fpr': ent_metric_1['micro_fpr'],
                'micro_fpr_no_empty': ent_metric_2['micro_fpr'],
            },
            'et': {
                'each_label': et_metric_1['each_label'],
                'micro_fpr': et_metric_1['micro_fpr'],
                'macro_fpr': et_metric_1['macro_fpr'],
                'micro_fpr_no_empty': et_metric_2['micro_fpr'],
                'macro_fpr_no_empty': et_metric_2['macro_fpr']
            }
        }

        return metrics

    def get_evaluate_event_fpr(self, results_pred, results_true, filter_empty=True):
        assert len(results_pred) == len(results_true)

        metric_result = dict()
        pred_event_num, true_event_num, correct_event_num = 0., 0., 0.
        pred_num_each_et, true_num_each_et, correct_num_each_et = defaultdict(int), defaultdict(int), defaultdict(int)

        for pe, te in zip(results_pred, results_true):
            p_ent, p_et = pe[:3], pe[-1]
            t_ent, t_et = te[:3], te[-1]

            if filter_empty:
                if pe == ('', -1, -1, 'Non-event') and te == ('', -1, -1, 'Non-event'):
                    continue
                if pe != ('', -1, -1, 'Non-event'):
                    pred_event_num += 1
                    pred_num_each_et[p_et] += 1
                if te != ('', -1, -1, 'Non-event'):
                    true_event_num += 1
                    true_num_each_et[t_et] += 1
            else:
                pred_event_num += 1
                true_event_num += 1
                pred_num_each_et[p_et] += 1
                true_num_each_et[t_et] += 1
            if pe == te:
                correct_event_num += 1
                correct_num_each_et[p_et] += 1

        p_micro = correct_event_num / (pred_event_num + 1e-10)
        r_micro = correct_event_num / (true_event_num + 1e-10)
        f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-10)

        metric_result['micro_fpr'] = f'{round(f1_micro, 3)}, {round(p_micro, 3)}, {round(r_micro, 3)}'

        ets = true_num_each_et.keys()  # 事件类型
        p_macro, r_macro, f1_macro = 0., 0., 0.
        metric_result['each_label'] = dict()
        for k in ets:
            metric_result['each_label'][k] = dict()
            p = correct_num_each_et[k] / (pred_num_each_et[k] + 1e-10)
            r = correct_num_each_et[k] / (true_num_each_et[k] + 1e-10)
            f1 = 2 * p * r / (p + r + 1e-10)
            metric_result['each_label'][k]['fpr'] = f'{round(f1, 3)}, {round(p, 3)}, {round(r, 3)}'
            metric_result['each_label'][k]['pn-tn-cn'] = f'{pred_num_each_et[k]}, {true_num_each_et[k]}, {correct_num_each_et[k]}'

            p_macro += p
            r_macro += r
            f1_macro += f1

        p_macro = p_macro/len(ets)
        r_macro = r_macro/len(ets)
        f1_macro = f1_macro/len(ets)
        metric_result['macro_fpr'] = f'{round(f1_macro, 3)}, {round(p_macro, 3)}, {round(r_macro, 3)}'
        return metric_result

    def get_evaluate_et_fpr(self, results_pred, results_true, filter_empty=True):
        assert len(results_pred) == len(results_true)

        # et2id, id2et = get_schema()
        # pred_ets = np.array([et2id[et[-1]] for et in results_pred])
        # true_ets = np.array([et2id[et[-1]] for et in results_true])

        # metric_result_sklearn = self.compute_metrics('classification', true_ets, pred_ets, labels=list(et2id.keys()))

        metric_result = dict()

        pred_et_num, true_et_num, correct_et_num = 0., 0., 0.
        pred_num_each_et, true_num_each_et, correct_num_each_et = defaultdict(int), defaultdict(int), defaultdict(int)

        for pe, te in zip(results_pred, results_true):
            p_et = pe[-1]
            t_et = te[-1]

            if filter_empty:
                if p_et == 'Non-event' and t_et == 'Non-event':
                    continue
                if p_et != 'Non-event':
                    pred_et_num += 1
                    pred_num_each_et[p_et] += 1
                if t_et != 'Non-event':
                    true_et_num += 1
                    true_num_each_et[t_et] += 1
            else:
                pred_et_num += 1
                true_et_num += 1
                pred_num_each_et[p_et] += 1
                true_num_each_et[t_et] += 1
            if p_et == t_et:
                correct_et_num += 1
                correct_num_each_et[p_et] += 1

        p_micro = correct_et_num / (pred_et_num + 1e-10)
        r_micro = correct_et_num / (true_et_num + 1e-10)
        f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-10)

        metric_result['micro_fpr'] = f'{round(f1_micro, 3)}, {round(p_micro, 3)}, {round(r_micro, 3)}'

        ets = true_num_each_et.keys()  # 事件类型
        p_macro, r_macro, f1_macro = 0., 0., 0.
        metric_result['each_label'] = dict()
        for k in ets:
            metric_result['each_label'][k] = dict()
            p = correct_num_each_et[k] / (pred_num_each_et[k] + 1e-10)
            r = correct_num_each_et[k] / (true_num_each_et[k] + 1e-10)
            f1 = 2 * p * r / (p + r + 1e-10)
            metric_result['each_label'][k]['fpr'] = f'{round(f1, 3)}, {round(p, 3)}, {round(r, 3)}'
            metric_result['each_label'][k]['pn-tn-cn'] = f'{pred_num_each_et[k]}, {true_num_each_et[k]}, {correct_num_each_et[k]}'

            p_macro += p
            r_macro += r
            f1_macro += f1

        p_macro = p_macro/len(ets)
        r_macro = r_macro/len(ets)
        f1_macro = f1_macro/len(ets)
        metric_result['macro_fpr'] = f'{round(f1_macro, 3)}, {round(p_macro, 3)}, {round(r_macro, 3)}'
        # print(metric_result_sklearn)
        # print(metric_result)
        return metric_result

    def get_evaluate_ent_fpr(self, results_pred, results_true, filter_empty=True):
        assert len(results_pred) == len(results_true)

        metric_result = dict()
        pred_ent_num, true_ent_num, correct_ent_num = 0., 0., 0.

        for pe, te in zip(results_pred, results_true):
            p_ent = pe[:3]
            t_ent = te[:3]
            if filter_empty:
                if p_ent == ('', -1, -1) and t_ent == ('', -1, -1):
                    continue
                if p_ent != ('', -1, -1):
                    pred_ent_num += 1
                if t_ent != ('', -1, -1):
                    true_ent_num += 1
            else:
                pred_ent_num += 1
                true_ent_num += 1
            if p_ent == t_ent:
                correct_ent_num += 1

        p_micro = correct_ent_num / (pred_ent_num + 1e-10)
        r_micro = correct_ent_num / (true_ent_num + 1e-10)
        f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-10)
        metric_result['micro_fpr'] = f'{round(f1_micro, 3)}, {round(p_micro, 3)}, {round(r_micro, 3)}'

        return metric_result

    def get_et_label_fpr(self, y_true, y_pred):
        print('*'*30)
        print(y_true[:10])
        print(y_pred[:10])
        print('*'*30)

        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')

        f1_micro = f1_score(y_true, y_pred, average='micro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        accuracy = (y_true == y_pred).mean()

        return accuracy, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def compute_metrics(self, output_mode, y_true, y_pred, labels=None):
        print('*'*30)
        print(y_true[:10])
        print(y_pred[:10])
        print('*'*30)

        metric = dict()

        for metric_mode in ['macro', 'micro']:
            p, r, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average=metric_mode)
            metric[f'{metric_mode}_fpr'] = f'{round(f1, 3)}, {round(p, 3)}, {round(r, 3)}'

        metric_each_label = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
        metric['each_label'] = dict()
        if not labels:
            labels = sorted(set(y_true))
        for idx, label in enumerate(labels):
            metric['each_label'][label] = dict()
            metric['each_label'][label]['prf'] = [str(metric_each_label[j][idx]) for j in range(4)]

        if output_mode == 'classification':
            accuracy = (y_true == y_pred).mean()
        if output_mode == 'multi-label-classification':
            correct_num = 0
            for i in range(y_true.shape[0]):
                if (y_true[i] == y_pred[i]).all():
                    correct_num += 1
            accuracy = correct_num / y_true.shape[0]

        metric['accuracy'] = str(accuracy)
        return metric
