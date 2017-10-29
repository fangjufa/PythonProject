import numpy as np

def L_i_Vectorized(x,li,W):
    '''
    MultyClass SVM Loss,li = sum(max(0,s[j] - s[li] + 1)).It's more simple than mine,and more mathmatics.
    Here x is a data instance,li is the class label of this x data.W is Weight
    '''
    scores = np.dot(x,W)
    margines = np.maximum(0,scores - scores[li] + 1)
    margines[li] = 0
    return np.sum(margines)


