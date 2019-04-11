""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = counter= qas_old = 0
    for para in dataset:
        paragraph_text = para['context.tokens']
        qas_id = para['_id']
        if qas_id != qas_old:
            qas_old = qas_id
            counter = 0
            qas_id = qas_id + '_' + str(counter)
        else:
            counter +=1
            qas_id = qas_id + '_' + str(counter)
                
        question_text = para['question.tokens']
        orig_answer_text = para['answers'][0]
        start_position = para['answers'][1][0][0]
        end_position = para['answers'][1][0][1]
            
        total += 1
        if qas_id not in predictions:
            message = 'Unanswered question ' + qas_id + \
                              ' will receive score 0.'
            print(message)
            continue
        #ground_truths = [orig_answer_text]
        ground_truths = para['ground_truths']
        prediction = predictions[qas_id]
        exact_match_eg = metric_max_over_ground_truths(exact_match_score,
                                                     prediction, ground_truths)
        f1_eg = metric_max_over_ground_truths( f1_score, prediction, ground_truths)
        exact_match += exact_match_eg
        f1 += f1_eg
        if total < 20:
       	    print('############### Examples #####################')
            print("       Id         = ", qas_id)
            print("  Question text   = ",question_text)
            print(" Original Answer1 = ", ground_truths[0])
            print(" Original Answer2 = ", ground_truths[1])
            print("  Predicted Ans   = ", prediction)
            print("   Exact_match    = ", bool(exact_match_eg))
            print("    F1 score      =", f1_eg) 
       
    print("Total examples checked = ", total)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Evaluation for NarrativeQA ')
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset = json.load(dataset_file)
        
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
        
    print(json.dumps(evaluate(dataset, predictions)))
