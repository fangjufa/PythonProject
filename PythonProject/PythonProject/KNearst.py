import numpy as np

class NearstNeighbor:
    """ NearstNeighbor,o(1) in training,but o(N) in predict.And it's not good."""
    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    def train(this,X,label):
        """ train function.X is NxD data,where each row is an example.And label is N dimension array,which is the label of each example. """
        this.Xtr = X
        this.label = label

    def predict(this,X,K = 1):
        count = X.shape[0]
        #It's important.
        label = np.zeros(count,dtype = this.label.dtype)
        result = np.zeros(K)
        for i in range(count):
            '''Every row of Xtr minus row of i to get distance '''
            #Sum every element of a row
            distance = np.sum(np.abs(this.Xtr - X[i,:]),axis = 1)
            #Sort this array,and pick first K elements.
            #Because they're nearst ones. 
            distance = np.sort(distance,)
            result = distance[0:K]

            #There may be more than one label of the K elements,so pick the most one.
            temp = np.zeros(K)
            for j in range(K):
                k = j+1
                while k < K:
                    if result[j] == result[k]:
                        temp[j] += 1
            #Just when K = 1
            #label[i] = this.label[np.argmin(distance)]
            label[i] = this.label[result[np.argmax(temp)]]

        return label
