#https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/

#Q learnign creates a large "Q Table" which contains many values 
#We can then use this table to look up Q values for each scenario
#When created this table is random
#At the start the agent explores and updates those values in the table, thus getting better at the game

#All combinations ar eon the left size with actions at the top
#Inside the table is the q value of each state and action
#The agent will find the state and perform whichever action has the highest q value
#https://miro.medium.com/max/384/1*RhO9Ulh5nF_zc2pcgBHzAg.png

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
state = env.reset()


LEARNING_RATE = 0.1
#Measure of how important future actions are
DISCOUNT = 0.95
#How many times the agent runs
EPISODES = 2000

SHOW_EVERY = 100

#Posiiton varies between 0.6 and -1.2
#Velocity varies between 0.07 and -0.07
print(env.observation_space.high) #[0.6  0.07]
print(env.observation_space.low) #[-1.2  -0.07]

#Discrete observation size (Q Table size)
#use * len(env.observation_space.high) as some tables will need more than 2 
#columns as they have more variables in their states than just 2
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

#Between 0 and 1
#Higher epsilon the more likely to perform a random action (exploratory)
#Forces the model to explore more and discover new methods
epsilon = 0.5
START_EPSILON_DECAYING = 1
#// divides to an integer
END_EPSILON_DECAYING = EPISODES // 2
#How much epsilon decays per episode
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING / START_EPSILON_DECAYING)

#Table of size every possible action state * number of actions availible
#With values uniformly between -2 and 0
#Over time these values will be tweaked
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


ep_rewards = []
#Dictionary of episode num, average, worst and best
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

#quick helper-function that will convert our environment "state," which currently 
#contains continuous values that would wind up making our Q-Table gigantic and 
#take forever to learn.... to a "discrete" state instead
def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
	episode_reward = 0
	if episode % SHOW_EVERY == 0:
		render = True
		print(episode)
	else:
		render = False
	discrete_state = get_discrete_state(env.reset())
	done = False
	while not done:

		if np.random.random() > epsilon:
			# Get action from Q table
			action = np.argmax(q_table[discrete_state])
		else:
			# Get random action
			action = np.random.randint(0, env.action_space.n)
		
		
		#State is what we sense from the environment
		#In mountain cart case it is two values, position and velocity
		#These values only matter to us and not to the agent
		new_state, reward, done, _ = env.step(action)
		
		episode_reward += reward

		#Get new discrete state from new state
		new_discrete_state = get_discrete_state(new_state)
		
		#print(reward, new_state)
		if render:
			env.render()

		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]
			
			#https://pythonprogramming.net/static/images/reinforcement-learning/new-q-value-formula.png
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			
			#Update Q Table with new Q value based on action just taken
			q_table[discrete_state + (action, )] = new_q
		elif new_state[0] >= env.goal_position:
			q_table[discrete_state + (action, )] = 0
			print(f"MADE IT ON EPISODE {episode}")
		discrete_state = new_discrete_state
	
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

	ep_rewards.append(episode_reward)
	if not episode % SHOW_EVERY:
		np.save(f"qtables/{episode}-qtable.npy", q_table)
		average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
		aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

		print(f"Episode: {episode} Average: {average_reward} Min: {min(ep_rewards[-SHOW_EVERY:])} Max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=1)
plt.show()
