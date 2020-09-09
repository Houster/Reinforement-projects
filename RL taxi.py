import numpy as np
import gym
import matplotlib.pyplot as plt
import random
from tqdm import *
import pickle

#hyperparameters
ALPHA=0.02
EPSILON=[0.4,0.3,0.2,0.03]
N_EPISODES=5000

#No discount rate as we are using Monte carlo and not MDP

#Helper methods
#Epsilon greedy?
def policy(state, eps, qtable):
    if random.random() > eps:
        #take the greedy action
        return np.argmax(qtable[state])
    return np.random.choice([0,1,2,3,4,5])

#new average = old average + (gain - old_ave) *alpha
def update(ro,g,qtable):
    st,act=ro    
    qtable[st][act]+= (ALPHA) * (g-qtable[st][act])
    
def epsilon(ep,num_ep,schedule):
    num_eps=num_ep//len(schedule)
    num_eps+= 1 if num_eps % len(schedule) else 0
    idx=ep//num_eps
    return schedule[idx]
    
    
#Create a q table
qtable=np.zeros(shape=(500,6))

#Create environment
env=gym.make("Taxi-v3")

#print(env.reset())
#print('observation space: ',env.observation_space)
#print('action: ',env.action_space)

success=0
failed=0
scorecard=[]

for ep in tqdm(range(N_EPISODES)):
    state=env.reset()
    stop=False
    rollout=[]
    rewards=[]
    
    #policy evaluation
    while not stop:
        #Sample an action from your policy
        #equiprobable
        action=policy(state,epsilon(ep,N_EPISODES,EPSILON),qtable)
        new_state,reward,stop,_=env.step(action)
        
        #Save the triple
        rollout.append((state,action))
        rewards.append(reward)
        
        state=new_state
    
    if rewards[-1]==--1 or rewards[-1]==-10:
        failed+=1
        scorecard.append(0)
    else:
        success+=1
        scorecard.append(1)
      
        
    #policy improvement (Monte carlo first visit)
    visited=set()
    
    for i in range(len(rollout)):
       
        #check if we have seen (st,act)
        if rollout[i] in visited:
            continue
        
        visited.add(rollout[i])
        gain=sum(rewards[i:])
        
        update(rollout[i],gain,qtable)

print('success: ', (success/N_EPISODES) * 100)
print('failed: ', (failed/N_EPISODES) * 100)

plt.figure(figsize=(10,10))
plt.plot(scorecard),plt.show()

with open('taxi.pickle','wb') as f:
    pickle.dump(qtable, f)
    