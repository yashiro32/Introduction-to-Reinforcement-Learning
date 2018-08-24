import numpy as np 
import sys, os

from blackjack import Blackjack
from utils import load_data

if len(sys.argv) < 3:
    print('Missing Q-values')
    sys.exit(-1)

def greedy(state, q_values):
   return np.argmax(state_action_values(state, q_values))

def state_action_values(state, q_values):
   dealer, player, usable_ace = state
   return q_values[dealer, player, usable_ace]

prefix = sys.argv[1]
iterations = int(sys.argv[2])

print('Using %s, iteration: %d\n\n' %(prefix, iterations))

q_values = load_data(prefix=prefix, iterations=iterations)

win = 0
lose = 0
draw = 0
busted = 0

games = 2000

plays = []

for _ in range(games):

   blackjack = Blackjack(1000)
   state = blackjack.reset()

   terminate = False
   action = -1
   reward = -1

   print('Dealer top card: %d\n' %state[0])

   single_game = []

   while not terminate:
      state_action = state_action_values(state, q_values)
      action = greedy(state, q_values)

      print('Player: cards: %s, value: %d, Usable ACE: %s' %(str(blackjack.player_cards()), state[1], str(blackjack.has_usable_ace())))
      print('\tstate action: ', end='')
      for k, v in enumerate(state_action):
         print('(%s: %0.7f)' %(blackjack.to_action(k), v), end=', ')
      print('\n\taction: %s' %blackjack.to_action(action))

      state, reward, terminate = blackjack.step(action)

      single_game.append((state, action, reward))

   print('\nFinal')
   print('\tDealer: cards: %s, value: %d' %(str(blackjack.dealer_cards()), blackjack.dealer_hand()))
   print('\tPlayer: cards: %s, value: %d' %(str(blackjack.player_cards()), blackjack.player_hand()))

   plays.append(single_game)

   print()

   if 1 == reward:
      win += 1
      print('!!! WIN !!!')

   elif -1 == reward:
      if blackjack.player_hand() > 21:
         print('xxx BUSTED xxx: ', end=' ')
         busted += 1
      else:
         print('xxx LOSE xxx: ', end=' ')
         lose += 1

      print('%s\n' %str(blackjack.player_cards()))

   else:
      draw += 1
      print('DRAW!')

   print('\n=================================\n')

print('win: %d, lose: %d, busted: %d, draw: %d' %(win, lose, busted, draw))
print('win: %0.2f%%, lose: %0.2f%%, busted: %0.2f%%, draw: %0.2f%%' %(win/games * 100, lose/games * 100, busted/games * 100, draw/games * 100))