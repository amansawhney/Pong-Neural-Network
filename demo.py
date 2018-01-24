import numpy as np
import cPickle as pickle
import gym

h = 200
batch_size = 10
learning_rate = 1e-4 #slow to converge
gamma = 0.99 #discount factor - optimizing for the short-term rewards
decay_rate = 0.99 #gradient decent
#hardmode is on
resume = False

#init model
D = 80*80

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    #xavier initialization - taking in acount the hidden layer
    #key value pair
    #low level
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  #makes weights not to small and not to big so it can change nicely
    model['W2'] = np.random.randn(H) / np.sqrt(H)  #makes weights not to small and not to big so it can change nicely
    # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
grad_buffer = { k: np.zeros_like(v) for k,v in model.iteritems()} #stores the gradient
rmsprop_cache = {k: np.zeros_like(v) for k,v in model.iteritems()} #stores the value of a formula which is a set constent

#activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) #squashing
def prepro(I):
    I = I[35:195] #croping the image
    I = I[::2, ::2, 0] #downsample
    I =[I==144] = 0 #erase background
    I =[I=109] = 0
    I[I !=0] = 1 #everything else is set to 1
    return I.astype(np.float).ravel() #flatten

def discount_reward(r):
    discount_r = np.zeros_like(r)
    running_add = 0 #storing rewards
    for t in reversed(xrange(0, r.size))
        if r[t] !=0: running_add = 0
        running_add = running_add * gamma + r[t]
        discount_r[t] = running_add
    return discount_r