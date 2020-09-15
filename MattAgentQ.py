"""
This is the template for the tic-tac-toe assessment
"""

import pickle
import numpy as np
import random

# TODO add other imports here

# TODO rename TemplateAgent to another class name
class QAgent():

   def __init__(self, symbol, num_episodes = 100):
      # do not remove or change these variables
      self.symbol = symbol
      self.challenge = False

      # TODO add your variables below
      self.ALPHA=0.1
      self.EPSILON=1
      self.DECAY=0.995
      self.GAMMA=0.95
      self.weights=np.ones((9,9))
  
   # Called after challenge_mode() is called to prepare the agent.
   # If it is challenge mode, then load the agent's tables/weights
   def prepare(self):
      if self.challenge:
         # TODO load the data
         with open("AgentQ.pickle",'rb') as f:
             weights=pickle.load(f)        
         return

   def q_value(self, state,action,weights):
       state=np.array(state) 
       dot=np.dot(weights,state)
       #print('dot is: ',dot)
       if action is None:
            return dot
       return dot[action]
   
    


# action to take in the state, return the action that the agent takes
   # if it is challenge mode, then  the agent should no longer 'learn'
   def action(self, state, episode):
      state=np.array(state)
      avail_moves= [ move for move, num in enumerate(state) if num==0]
      
      if self.challenge:
          action=np.argmax(self.weights[state])
          return action
      
      else:

         dot=np.dot(self.weights,state)
         #print("dot is: ", dot)
         
         if random.random()>self.EPSILON:
             
             remaining_moves=[dot[x] if x in avail_moves else -99**999 for x in range(len(dot))]
             #print('remaining_moves is: ', np.argmax(remaining_moves)
             #return np.random.choice(avail_moves)
             return np.argmax(remaining_moves)
         
         else:

             return np.random.choice(avail_moves)
          

      # TODO perform policy evaluation
    
      

      #return np.random.randint(0, 9)

   # the reward resulting from the action taken when the agent is in state
   # the action results in a new_state and gets the one step reward
   # new_state is None when episode ends
   # update will not be called if it is challenge mode
   def update(self, state, action, reward, new_state):       
        
      
        
       if new_state is None:
           td_target = reward
           
       else:
           state=np.array(state)    
           self.EPSILON = self.DECAY * self.EPSILON
    
          
    
           #SARSA here
           #new_action=self.action(new_state, episode=0)
           #td_target= reward + self.GAMMA *(self.q_value(new_state,new_action,self.weights))
           #max action for Q learning update
           max_action=np.argmax(self.q_value(state,None,self.weights))
           td_target= reward + self.GAMMA *(self.q_value(new_state,max_action,self.weights))
           
           #latest error here
           #print("td target is: ", td_target, " and state is: ", state)
           #print("Q-value is: ", self.q_value(state,action,self.weights))
           #print("weights is: ", self.weights)
           
       state=np.array(state)
       
       td_error=(td_target-self.q_value(state,action,self.weights)) * state
       #print('td error is: ', td_error)
       #print("td_target is: ", td_target, " q value is: ",self.q_value(state,action,self.weights), " and state is: ", state)
       self.weights[action]+= self.ALPHA * td_error
       #print('weights are: ', self.weights)
       
      #pass

   # give your agent a name
   def name(self):
      # TODO give your agent a name
      return 'Agent Q here'

   # called before closing the agent
   # should save any data
   def close(self):
      if self.challenge:
         return

      # TODO save tables/weights using pickle
      with open('AgentQ.pickle','wb') as f:
          pickle.dump(self.weights, f)

   def __str__(self):
      return str(self.symbol)

