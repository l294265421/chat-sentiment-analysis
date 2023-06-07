import json
import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from chat_sentiment_analysis.utils.prompter import Prompter
from chat_sentiment_analysis.common import common_path
from chat_sentiment_analysis.utils import file_utils


def parse(task_name: str, label: str):
    """

    :param task_name:
    :param true:
    :return:
    """
    result = []
    if task_name == 'extract aspect terms from the sentence':
        # there are no aspect terms in the sentence.
        if label != 'there are no aspect terms in the sentence.':
            result = label.split(', ')
    elif task_name == 'extract opinion term from the sentence':
        # there are no opinion terms in the sentence.
        if label != 'there are no opinion terms in the sentence.':
            result = label.split(', ')
    elif task_name == 'extract aspect term-opinion term pairs from the sentence':
        # there are no aspect term-opinion term pairs in the sentence.
        if label != 'there are no aspect term-opinion term pairs in the sentence.':
            result = label.split('; ')
    elif task_name == 'extract aspect term, sentiment, opinion term triplets from the sentence':
        # there are no aspect term, sentiment, opinion term triplets in the sentence.
        if label != 'there are no aspect term, sentiment, opinion term triplets in the sentence.':
            result = label.split('; ')
    else:
        raise NotImplementedError(task_name)
    return result


def precision_recall_f1(pred: set, true: set):
    """

    :param pred:
    :param true:
    :return:
    """
    intersection = pred.intersection(true)
    precision = len(intersection) / len(pred)
    recall = len(intersection) / len(true)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate(instances):
    """

    :param instances:
    :return:
    """
    true = set()
    pred = set()
    for instance in instances:
        input = instance[0]
        true.add(['%s##%s' % (input, e) for e in instance[1]])
        pred.add(['%s##%s' % (input, e)for e in instance[2]])
    result = precision_recall_f1(pred, true)
    return result


def print_precision_recall_f1(metrics: dict):
    """

    :param metrics:
    :return:
    """
    precision, recall, f1 = metrics['precision'], metrics['recall'], metrics['f1']
    result = '\t'.join([','.join(precision), ','.join(recall), ','.join(f1)])
    print(result)


def main(
    data_path: str = os.path.join(common_path.data_dir, 'task_data', 'asote.test.json.with_pred')
):
    lines = file_utils.read_all_lines(data_path)
    task_and_instances = {}
    for line in lines:
        line_obj = json.loads(line)
        dataset = line_obj['dataset']
        task = line_obj['instruction']
        input = line_obj['input']
        output = parse(task, line_obj['output'])
        pred = parse(task, line_obj['pred'])
        key = f'{dataset}_{task}'
        if key not in task_and_instances:
            task_and_instances[key] = []
        task_and_instances[key].append([input, output, pred])

    for task, instances in task_and_instances.items():
        metrics = evaluate(instances)
        print(f'task: {task}')
        print_precision_recall_f1(metrics)


if __name__ == "__main__":
    fire.Fire(main)
