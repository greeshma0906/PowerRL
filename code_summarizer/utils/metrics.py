import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

smoothie = SmoothingFunction().method4
rouge = Rouge()


def compute_bleu(reference, predicted):
    return sentence_bleu([reference], predicted, smoothing_function=smoothie)


def compute_rouge(reference, predicted):
    try:
        ref_sentence = " ".join(reference)
        pred_sentence = " ".join(predicted)
        return rouge.get_scores(pred_sentence, ref_sentence, avg=True)
    except:
        return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}