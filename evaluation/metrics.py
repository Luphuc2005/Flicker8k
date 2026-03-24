import sys
import os

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def calculate_metrics(gts, res):
    """
    gts: {img_id: [{"caption": string}, ...]}
    res: {img_id: [{"caption": string}]}
    """
    # Tokenize
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # Scorers
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for s, m in zip(score, method):
                final_scores[m] = s
        else:
            final_scores[method] = score
            
    return final_scores
