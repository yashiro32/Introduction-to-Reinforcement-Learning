import os
import pickle as pickle

import numpy as np

make_name = lambda p, i: '%s_%i.pickle' %(p, i)

def min_max_scaler(st):
   d_s = lambda x: (x - 1)/9
   p_s = lambda x: (x - 1)/20
   u_s = lambda x: x / 2
   return (d_s(st[0]), p_s(st[1]), u_s(st[2]))

def load_data(*, prefix, iterations, root='data'):
   return pickle.load(open(os.path.join(root, make_name(prefix, iterations)), 'rb'))

def save_data(*, data, prefix, iterations, root='data'):
   fn = os.path.join(root, make_name(prefix, iterations))
   pickle.dump(data, open(fn, 'wb'))
   return fn

def to_matrix(q_values):
   states = np.zeros((11, 22, 3, 4), dtype=float)
   for d in range(1, 11):
      for p in range(1, 22):
         for a in range(3):
            states[d, p, a] = q_values[(d, p, a)]
   return states

class QValue:

   def __init__(self):
      self.q_values = {}
      self.patch()

   def __getitem__(self, key):
      #just state only
      if len(key) == 3:
         return self.q_values[key] if key in self.q_values else [0]
      #assume state action pair
      st, ac = key
      if ac < 0:
         return 0
      return self.q_values[st][ac]

   def __setitem__(self, key, value):
      st, ac = key
      if ac >= 0:
         self.q_values[st][ac] = value

   def patch(self):
      count = 0
      for d in range(1, 11):
         for p in range(1, 22):
            # 3 state: no usable ace, usable ace, ace in play
            for u in range(0, 3): 
               if (d, p, u) not in self.q_values:
                  count += 1
                  self.q_values[(d, p, u)] = [0, 0, 0, 0]
      return count

   def dump_values(self):
      return self.q_values

class Stats:

   def __init__(self, N0 = 100):
      self.state = {}
      self.state_action = {}
      self.N0 = N0
      print('N0 set to %d' %self.N0)

   def reset_traces(self):
      self.trace = {}

   def inc_trace(self, st, ac):
      self.inc_counter((st, ac), self.trace)

   def decay_trace(self, st, ac, gam, lam):
      self.trace[(st, ac)] *= gam * lam
      pass

   def get_trace(self, st, ac):
      return self.trace[(st, ac)]

   def get_counter(self, st, acc):
      return acc[st] if st in acc else 0

   def inc_counter(self, st, acc):
      if st not in acc:
         acc[st] = 0
      acc[st] += 1

   def update_stats(self, st, ac):
      self.inc_state(st)
      self.inc_state_action(st, ac)
      self.inc_trace(st, ac)

   def inc_state(self, st):
      self.inc_counter(st, self.state)

   def inc_state_action(self, st, ac):
      self.inc_counter((st, ac), self.state_action)

   def epsilon(self, st):
      return self.N0 / (self.N0 + self.get_counter(st, self.state))

   def alpha(self, st, ac):
      return 1 / self.get_counter((st, ac), self.state_action)

def inc_counter(state, counter):
   if state not in counter:
      counter[state] = 0
   counter[state] += 1

def print_policy(states, mod=4):
   print('\n\n')
   for i in range(len(states)):
      if 0 == (i % mod):
         print()
         print('  ', end='')

      v = np.argmax(states[i])
      if 0 == v:
         g = '\u2190'
      elif 1 == v:
         g = '\u2193'
      elif 2 == v:
         g = '\u2192'
      else:
         g = '\u2191'
      print(g, end='')

   print('\n')