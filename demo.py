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


