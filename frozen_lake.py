import sys, gym
import numpy as np

#from utils import print_policy

episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 10

"""
[[0, 3, 0, 5], [0, 0, 0, 0]]
"""

# Create the game.
env = gym.make('FrozenLake-v0')

q_values = [ np.zeros(env.action_space.n) for _ in range(env.observation_space.n) ]
def Q(st, act=None):
	if act is None:
		return q_values[st]
	return q_values[st][act]

#print(len(q_values[0]))

#exploration
epsilon = 0.01
# discounting
gamma = 1.0
# learning rate
alpha = 0.01

def policy(st):
	if np.random.rand() > epsilon:
		return np.argmax(Q(st))
	return np.random.randint(0, env.action_space.n)

win, lose = 0, 0

#print('[H[J')

for e in range(episodes):
	#initialize.
	state = env.reset()
	action = policy(state)

	terminate = False

	while not terminate:
		#print('[1;1H')

		env.render()

		new_state, reward, terminate, _ = env.step(action)

		if terminate:
			if reward == 0:
				lose += 1
				reward = -10
			else:
				win += 1
			tf_target = reward

		else:
			new_action = policy(new_state)
			#print("New action: ", new_action)
			tf_target = reward + gamma * Q(new_state, new_action)

		td_error = tf_target - Q(state, action)

		q_values[state][action] = q_values[state][action] + (alpha * td_error)

		state = new_state
		action = new_action
	
	#print('\n\nEpisodes: %d, WIN: %d, LOSE: %d' % (e, win, lose))

#print(q_values)

