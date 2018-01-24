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

