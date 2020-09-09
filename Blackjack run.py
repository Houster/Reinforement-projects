import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import *
import pickle

with open("Blackjack.pickle",'rb') as f:
    qtable=pickle.load(f)
    
N_EPISODES=2000
CAPITAL=100

winnings=[]

env=gym.make("Blackjack-v0")


def ensure(state,qtable):
    if state not in qtable:
        qtable[state]=np.zeros(2)
        
def policy(state, qtable):
    ensure(state,qtable)
    return np.argmax(qtable[state])

normalize_state=lambda st: (st[0], st[1], 1 if st[2] else 0)

for _ in range(N_EPISODES):
    state=normalize_state(env.reset())
    stop=False
    
    while not stop:
        action=policy(state,qtable)
        new_state,reward,stop,_=env.step(action)
        
        if stop:
            if reward==1:
                CAPITAL+=5
            elif reward==-1:
                CAPITAL-=5
            winnings.append(CAPITAL)
        else:
            state=normalize_state(new_state)
    
    if CAPITAL <0:
        break
    
plt.plot(winnings)
plt.title("Winnings given $100 capital")