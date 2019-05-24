import numpy as np
from tqdm import tqdm
import seaborn as sns
import time as t

sns.set_style("darkgrid")
# %pylab inline
import random

starting_pos = [0, 3]

def slip(action, initState):
    finalState = np.array(initState) + np.array(action)
    while list(finalState) not in terminationStates:
        if -1 in list(finalState) or gridDim in list(finalState):
            finalState = initState
            return finalState
        initState = finalState
        finalState = np.array(initState) + np.array(action)
    return finalState


#####################MONTE-CARLO POLICY EVALUATION


gamma = 0.9  # We set the discount rate at 0.9
gridDim = 4
terminationStates = [[1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [1, 1], [gridDim - 1, 0]] # Cracks and endgoal
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]] #left, right, down, up
numIterations = 10000    #we've set the number of iterations at 10000 for this purpose

#make the matrices to be filled and the matrix representative of the environment
V = np.zeros((gridDim, gridDim))
returns = {(i, j): list() for i in range(gridDim) for j in range(gridDim)}
deltas = {(i, j): list() for i in range(gridDim) for j in range(gridDim)}
states = [[i, j] for i in range(gridDim) for j in range(gridDim)]


def generateEpisode():
    initState = starting_pos
    episode = []
    rewardSize = 0
    while True:
        if list(initState) in terminationStates:
            return episode
        action = random.choice(actions)
        slip_chance = [0.05, 0.95]
        resultslip = np.random.choice(2, 1, p=slip_chance)
        if resultslip == [0]:
            finalState = slip(action, initState)
        else:
            finalState = np.array(initState) + np.array(action)

        if -1 in list(finalState) or gridDim in list(finalState):
            finalState = initState
        # print(finalState)

        if (finalState[0] == 1 and finalState[1] == 3) or (finalState[0] == 2 and finalState[1] == 3) or (
                finalState[0] == 3 and finalState[1] == 3) or (finalState[0] == 3 and finalState[1] == 2) or (
                finalState[0] == 3 and finalState[1] == 1) or (finalState[0] == 1 and finalState[1] == 1):
            rewardSize = -10

        if (finalState[0] == 2 and finalState[1] == 2):
            rewardSize = 20

        if finalState[0] == 3 and finalState[1] == 0:
            rewardSize = 100
        episode.append([list(initState), action, rewardSize, list(finalState)])
        initState = finalState

t1 = t.time()

for it in tqdm(range(numIterations)):
    episode = generateEpisode()
    G = 0
    #print(episode)
    #go through the steps of the episodes
    for i, step in enumerate(episode[::-1]):
        G = gamma * G + step[2]
        #using the monte-carlo algorithm where averages are calculated based on the returns for the state values
        if step[0] not in [x[0] for x in episode[::-1][len(episode) - i:]]:
            statemove = (step[0][0], step[0][1])
            returns[statemove].append(G)
            newValue = np.average(returns[statemove])
            deltas[statemove[0], statemove[1]].append(np.abs(V[statemove[0], statemove[1]] - newValue))
            V[statemove[1], statemove[0]] = newValue

#final values
print(V)

t2 = t.time()
print("Time elapsed:",t2-t1)