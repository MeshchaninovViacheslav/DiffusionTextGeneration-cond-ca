import numpy as np
from estimation_utils.pymteval import BLEUScore

def bleu_score(prediction, reference):
    scorer = BLEUScore(smoothing=1.0)
    scorer.reset()
    scorer.append(prediction, [reference])
    return scorer.score()

def mbr(samples):

    score_dict = {}
    for idx_y in range(len(samples)):
        y = samples[idx_y]
        
        blue_lst = []
        for idx_x in range(len(samples)):
            if idx_x != idx_y:
                x = samples[idx_x]
                b = bleu_score(prediction=y, reference=x)
                blue_lst.append(b)
        score_dict[idx_y] = np.array(blue_lst).mean()

    best_y = sorted(score_dict.items(), key=lambda item: item[1])[-1]

    return samples[best_y[0]]