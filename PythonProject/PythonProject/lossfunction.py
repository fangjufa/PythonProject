import numpy as np
#from scipy.special import expit

class LossFuntion(object):

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    def A_test(self,str):
        print(str)

    def compute_score(this,W,x,b):
        #scores = np.zeros(b.size)
        #for i in range(b.size):
        #    scores[i] = np.dot(W[i],x) + b[i]
    
        #mul = np.dot(W,x)
        x = np.multiply(x,1/255)
        return np.dot(W,x) + b #  scores# mul + b

    def WW(self,W,b,x,lbli):
        """Weston watkins ways to calculate loss and gradients. """
        gradientW = np.zeros(W.shape)
        gradientb = np.zeros(b.size)
        gradientx = np.zeros(x.shape)
        s = self.compute_score(W,x,b)
        loss = 0

        for k in range(b.size):
            #WW ways to calculate gradients
            #if k equals to li,then there is no gradient of W and b.
            if k == lbli:
                continue
            lossk = np.max([0,s[k] - s[lbli] + 1])
            loss += lossk
            #if lossk less than 0,then there is also no gradient of W and b.
            if lossk > 0:
                gradientW[k] += x
                gradientb[k] += 1
                gradientW[lbli] -= x
                gradientb[lbli] -= 1
                gradientx += (W[k] - W[lbli])
        return gradientx,gradientW,gradientb,loss

    def SoftMax(self,W,b,x,lbli):
        """sotfmax ways to calculate loss and gradients"""
        size = b.size
        gradientW = np.zeros(W.shape)
        gradientb = np.zeros(b.size)
        loss = 0
        s = self.compute_score(W,x,b)
        p = np.exp(s)
        p /= np.sum(p)
        for i in range(size):
            if i == lbli:
                #Actually the loss function is loss = -np.sum(yi*log(p[i]))
                #we omit y here,cause we assume that y = 1 if k == li,else y = 0
                loss = -np.log(p[i])

                #here p[k]-1,actually it is p[k] - y[k]
                gradientW[i] += np.multiply(x,(p[i] - 1))
                gradientb[i] += p[i] - 1
            else:
                gradientW[i] += np.multiply(x,p[i])
                gradientb[i] += p[i] 

        return gradientW,gradientb,loss