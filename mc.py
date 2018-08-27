import sys, time
import numpy as np

from tqdm import *

from blackjack import Blackjack
from blackjack_biased import Blackjack_Biased
#from plotgrid import plot_grid
from utils import save_data, load_data, inc_counter

N0 = 100
HITS = Blackjack.HITS # 0
STICK = Blackjack.STICK # 1
USE_ACE = Blackjack.USE_ACE # 2
IDLE_ACE = Blackjack.IDLE_ACE # 3
DEFAULT_EPISODE = 3

#Number of episodes
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EPISODE
use_bias_env = True if len(sys.argv) > 2 else False

#initialize states [dealer_top_card, player_hand, usable_ace, actions]
#actions = 0 - HITS, 1 - STICK
#0 element not used
if use_bias_env:
   print('Initializing states with %s_%s.pickle' %(sys.argv[2], sys.argv[3]))
   states = load_data(prefix=sys.argv[2], iterations=int(sys.argv[3]))
else:
   states = np.zeros((11, 22, 3, 4), dtype=float)

state_counter = {}
state_action_counter = {}

def policy(st, N0, states, state_count):
   epsilon = N0 / (N0 + state_count)
   if np.random.uniform() > epsilon:
      return np.argmax(states[st[0], st[1], st[2], :])
   return np.random.randint(low=HITS, high=(IDLE_ACE + 1))

def update(*, state, action, states, state_action, gain):
   q_state_action = states[state[0], state[1], state[2], action]
   states[state[0], state[1], state[2], action] = q_state_action \
         + ((state_action ** -1) * (gain - q_state_action))

get_counters = lambda st, acc: acc[st] if st in acc else 0

def extract_decision(states, usable_ace):
   decision = np.zeros(shape=(len(states), len(states[0]), 1), dtype=np.int16)
   q_value = np.zeros(shape=(len(states), len(states[0]), 1))
      
   for i, d in enumerate(states):
      for j, p in enumerate(d):
         decision[i, j] = np.argmax(p[usable_ace])
         q_value[i, j] = max(p[usable_ace])

   return (decision, q_value)

print('EPISODES: %d ' %EPISODES)
print('Biased environment: ', use_bias_env)

win = 0
lose = 0
draw = 0
has_ace = 0
hits_counter = 0

for e in tqdm(range(EPISODES)):
   #Initialize environment
   episode = []
   blackjack = Blackjack_Biased(1000) if use_bias_env else Blackjack(1000)
   state = blackjack.reset()
   terminate = False

   #Sample an episode
   while not terminate:
      action = policy(state, N0, states, get_counters(state, state_counter))

      if action == Blackjack.HITS:
      	hits_counter += 1

      new_state, reward, terminate = blackjack.step(action)
      episode.append((state, action, reward, new_state))

      if terminate:
         break

      state = new_state

   has_ace += 1 if blackjack.has_usable_ace() else 0

   _, _, outcome, _ = episode[-1]

   if 1 == outcome:
      win += 1
   elif -1 == outcome:
      lose += 1
   else:
      draw += 1

   #Perform policy improvement

   for i, v in enumerate(episode):
      state, action, reward, _ = v

      inc_counter(state, state_counter)
      inc_counter((state, action), state_action_counter)

      gain = sum(ep[2] for ep in episode[i:])

      update(state=state, action=action, states=states, \
            state_action=get_counters((state, action), state_action_counter), gain=gain)


print('win: %d, lose: %d, draw: %d' %(win, lose, draw))
print('win: %0.1f%%, lose: %0.1f%%, draw: %0.1f%%' %(win/EPISODES * 100, lose/EPISODES * 100, draw/EPISODES * 100))
print('usable ace: %0.2f%%' %(has_ace/EPISODES * 100))
print('HITS: %0.2d' %(hits_counter))

fn = save_data(data=states, \
      prefix='montecarlo' + ('-biased' if use_bias_env else ''), iterations=EPISODES)
print('Saved %s' %fn)

#dealer = np.arange(0, 11)
#player = np.arange(0, 22)
#nua_decision, nua_value = extract_decision(states, 0)
#
#plot_grid(dealer=dealer, player=player, value=nua_value, decision=nua_decision)
