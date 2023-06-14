import numpy as np
import random
import pdb
import os

from sklearn.metrics.pairwise import cosine_similarity as cossim

def test_roc(embedding_list, label_list):
    test_embs = np.array(embedding_list)
    test_label = np.array(label_list)

    scores = cossim(test_embs, test_embs)
    label_mat = test_label.reshape([-1,1]) == test_label.reshape([1,-1])
    
    impostor = np.sum(1 - label_mat)
    genuine = np.sum(label_mat)
    
    negative_scores = scores[label_mat == False]
    positive_scores = scores[label_mat-np.identity(len(embedding_list)) == True]
    
    def score(threshold):
        accept = (scores > threshold).astype(np.float32)                
        FA = np.sum((label_mat - accept)<0)
        FR = np.sum((label_mat - accept)>0)
        return FA/impostor, FR/(genuine-len(embedding_list))
    
    high = 1
    low = -1
    EER, threshold = None, None

    while abs(high - low) > 1e-6:
        middle = (high + low)/2
        far, frr = score(middle)
        print(middle, far, frr)
        threshold = middle
        EER = far
        if frr > far:
            high = middle
        else:
            low = middle
    return EER, threshold, np.mean(negative_scores), np.mean(positive_scores)