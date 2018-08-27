import numpy as np

class Blackjack:

   HITS = 0
   STICK = 1
   USE_ACE = 2
   IDLE_ACE = 3

   def __init__(self, times = 5000):
      self.times = times

   def reset(self):
      self.dealer = []
      self.player = []
      self.deck = [10, 1, 2, 3, 10, 4, 5, 6, 10, 7, 8, 9, 10] * 4
      self.usable_ace = False
      np.random.shuffle(self.deck)
      for i in range(self.times):
         np.random.shuffle(self.deck)

      self.dealer.append(self.draw())
      self.player.append(self.draw())
      self.dealer.append(self.draw())
      self.player.append(self.draw())

      return self.make_return_state()

   def step(self, action):

      if Blackjack.HITS == action: 
         self.player.append(self.draw())
         reward = -1 if self.is_busted(self.player_hand()) else 0
         returns = 2 if self.has_morecards(self.player_cards(), 4) and self.is_busted(self.player_hand()) != True else reward # Receive more rewards if player has more cards on his deck.  

         return (self.make_return_state(), returns, reward < 0)

      elif Blackjack.USE_ACE == action:
         reward = -1
         if self.has_usable_ace() and not self.usable_ace:
            self.usable_ace = True
            reward = -1 if self.is_busted(self.player_hand()) else 0

         return (self.make_return_state(), reward, reward < 0)

      elif Blackjack.IDLE_ACE == action:
         reward = -1
         if self.has_usable_ace() and self.usable_ace:
            self.usable_ace = False
            reward = 0

         return (self.make_return_state(), reward, reward < 0)

      #stick

      player_hand = self.player_hand()
      player_busted = self.is_busted(player_hand)

      assert not player_busted

      #Player must get at least 15
      if player_hand < 15:
         return (self.make_return_state(), -1, True)

      while True:
         self.dealer.append(self.draw())

         total = sum(self.dealer)

         if (total < 17):
            continue

         if self.is_busted(total):
	         returns = 2 if self.has_morecards(self.player_cards(), 4) else 1
	         return (self.make_return_state(), returns, True)
	         #return (self.make_return_state(), 1, True)

         if total < player_hand:
	         returns = 2 if self.has_morecards(self.player_cards(), 4) else 1
	         return (self.make_return_state(), returns, True)
	         #return (self.make_return_state(), 1, True)

         if total > player_hand:
	         return (self.make_return_state(), -1, True)

         if total == player_hand:
	         returns = 2 if self.has_morecards(self.player_cards(), 4) else 0
	         return (self.make_return_state(), returns, True)
	         #return (self.make_return_state(), 0, True)


   def show_deck(self):
      return (self.deck)

   # dealer top card, your_hand, usable ace
   def make_return_state(self):
      ua = 1 if self.has_usable_ace() else 0
      ua += 1 if (ua and self.usable_ace) else 0
      return (self.dealer_top_card(), self.player_hand(), ua)

   def draw(self):
      return self.deck.pop(0)

   def dealer_top_card(self):
      return self.dealer[0]

   def player_cards(self):
      return self.player

   def dealer_cards(self):
      return self.dealer

   def player_hand(self):
      hand = sum(self.player)
      if self.has_usable_ace() and self.usable_ace:
         return hand + 10
      return hand

   def dealer_hand(self):
      return sum(self.dealer)

   def has_usable_ace(self):
      return 1 in self.player 

   is_busted = lambda self, value: value > 21

   def is_blackjack(self, owner):
	   return len(owner) == 2 and 10 in owner and 1 in owner

   # Check if the player has more than a certain number of cards.
   def has_morecards(self, owner, numcards):
	   return len(owner) > numcards

   def to_action(self, action):
      if Blackjack.HITS == action:
         return "HITS"
      elif Blackjack.STICK == action:
         return "STICK"
      elif Blackjack.USE_ACE == action:
         return "USE_ACE"
      elif Blackjack.IDLE_ACE == action:
         return "IDLE_ACE"
      else:
         return "ANON"

if __name__ == '__main__':
   bj = Blackjack(1000)
   state = bj.reset()

   print('deck = ', bj.show_deck())

   print('dealer top card: %d' %bj.dealer_top_card())
   print('player cards : %s' %str(bj.player_cards()))
   print('player hand : %d' %bj.player_hand())
   print('start state = %s' %str(state))

   terminate = False

   while not terminate:
      action = np.random.randint(0, 4)
      state, reward, terminate = bj.step(action)
      print('action = ', bj.to_action(action), ' next state = ', state, ' reward = ', reward)