#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
# add King's Move(the diagonal moves)
ACTION_UP_LEFT = 4
ACTION_UP_RIGHT = 5
ACTION_DOWN_LEFT = 6
ACTION_DOWN_RIGHT = 7
# add Stay
ACTION_STAY = 8

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

# state action pair value
actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP_LEFT, ACTION_UP_RIGHT, ACTION_DOWN_LEFT, ACTION_DOWN_RIGHT, ACTION_STAY]
stateActionValues = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(actions)))
startState = [3, 0]
goalState = [3, 7]

# set up destinations for each action in each state
actionDestination = []
for i in range(0, WORLD_HEIGHT):
    actionDestination.append([])
    for j in range(0, WORLD_WIDTH):
        destination = dict()
        destination[ACTION_UP] = [max(i - 1 - WIND[j], 0), j]
        destination[ACTION_DOWN] = [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
        destination[ACTION_LEFT] = [max(i - WIND[j], 0), max(j - 1, 0)]
        destination[ACTION_RIGHT] = [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
        destination[ACTION_UP_LEFT] = [max(i - 1 - WIND[max(j - 1, 0)], 0), max(j - 1, 0)]
        destination[ACTION_UP_RIGHT] = [max(i - 1 - WIND[min(j + 1, WORLD_WIDTH - 1)], 0), min(j + 1, WORLD_WIDTH - 1)]
        destination[ACTION_DOWN_LEFT] = [max(min(i + 1 - WIND[max(j - 1, 0)], WORLD_HEIGHT - 1), 0), max(j - 1, 0)]
        destination[ACTION_DOWN_RIGHT] = [max(min(i + 1 - WIND[min(j + 1, WORLD_WIDTH - 1)], WORLD_HEIGHT - 1), 0), min(j + 1, WORLD_WIDTH - 1)]
        destination[ACTION_STAY] = [max(min(i - WIND[j], WORLD_HEIGHT - 1), 0), j]
        actionDestination[-1].append(destination)

# play for an episode
def oneEpisode():
    # track the total time steps in this episode
    time = 0

    # initialize state
    currentState = startState

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        currentAction = np.random.choice(actions)
    else:
        currentAction = np.argmax(stateActionValues[currentState[0], currentState[1], :])

    # keep going until get to the goal state
    while currentState != goalState:
        newState = actionDestination[currentState[0]][currentState[1]][currentAction]
        if np.random.binomial(1, EPSILON) == 1:
            newAction = np.random.choice(actions)
        else:
            newAction = np.argmax(stateActionValues[newState[0], newState[1], :])
        # Sarsa update
        stateActionValues[currentState[0], currentState[1], currentAction] += \
            ALPHA * (REWARD + stateActionValues[newState[0], newState[1], newAction] -
            stateActionValues[currentState[0], currentState[1], currentAction])
        currentState = newState
        currentAction = newAction
        time += 1
    return time

# play for 500 episodes to make sure to get a more converged policy
# episodeLimit = 200

# figure 6.4
episodeLimit = 500
ep = 0
episodes = []
while ep < episodeLimit:
    time = oneEpisode()
    if time > 150:
        time = 150
    episodes.append(time)
    ep += 1

plt.figure()
plt.xlabel('Time steps')
plt.ylabel('Episodes with nine actions')
plt.plot(episodes)
plt.grid()
plt.show()

# display the optimal policy
optimalPolicy = []
for i in range(0, WORLD_HEIGHT):
    optimalPolicy.append([])
    for j in range(0, WORLD_WIDTH):
        if [i, j] == goalState:
            optimalPolicy[-1].append(' G')
            continue
        bestAction = np.argmax(stateActionValues[i, j, :])
        if bestAction == ACTION_UP:
            optimalPolicy[-1].append(' U')
        elif bestAction == ACTION_DOWN:
            optimalPolicy[-1].append(' D')
        elif bestAction == ACTION_LEFT:
            optimalPolicy[-1].append(' L')
        elif bestAction == ACTION_RIGHT:
            optimalPolicy[-1].append(' R')
        elif bestAction == ACTION_UP_LEFT:
            optimalPolicy[-1].append('UL')
        elif bestAction == ACTION_UP_RIGHT:
            optimalPolicy[-1].append('UR')
        elif bestAction == ACTION_DOWN_LEFT:
            optimalPolicy[-1].append('DL')
        elif bestAction == ACTION_DOWN_RIGHT:
            optimalPolicy[-1].append('DR')
        elif bestAction == ACTION_STAY:
            optimalPolicy[-1].append(' S')

for row in optimalPolicy:
    print(row)
print([" " + str(w) for w in WIND])

