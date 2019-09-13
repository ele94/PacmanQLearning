# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math,pickle

class Cuadrants:
    NORTHWEST = 'Northwest'
    NORTHEAST = 'Northeast'
    SOUTHWEST = 'Southwest'
    SOUTHEAST = 'Southeast'
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    CENTER = 'Center'

class NewQLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, extractor='MyFeatureExtractor', **args):
        "You can initialize Q-values here..."
        self.featExtractor = util.lookup(extractor, globals())()
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.QValueCounter = util.Counter()

        # uncomment to load existing qvalues
        # loading existing weights
        # self.loadTableFromFile('tables/gridnewfeatures.pkl')

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        stateFeatures = self.getFeaturesFromState(state)
        return self.QValueCounter[(stateFeatures, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return 0

        #stateFeatures = self.getFeaturesFromState(state)
        best_action = self.computeActionFromQValues(state)
        return self.getQValue(state, best_action)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return None

        best_action = None
        best_value = float('-inf')
        #stateFeatures = self.getFeaturesFromState(state)

        for action in self.getLegalActions(state):
            if self.getQValue(state, action) > best_value:
                best_value = self.getQValue(state, action)
                best_action = action
        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return None  # Terminal State, return None

        if self.epsilon > random.random():
            action = random.choice(legalActions)  # Explore
        else:
            action = self.computeActionFromQValues(state)  # Exploit

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        stateFeatures = self.getFeaturesFromState(state)
        nextStateFeatures = self.getFeaturesFromState(nextState)

        best_action = self.computeActionFromQValues(nextState)
        self.QValueCounter[(stateFeatures, action)] = ((1 - self.alpha) * self.getQValue(state, action) +
                                               self.alpha * (reward + self.discount * self.getQValue(nextState,
                                                                                                     best_action)))

    def getPolicy(self, state):
        stateFeatures = self.getFeaturesFromState(state)
        return self.computeActionFromQValues(stateFeatures)

    def getValue(self, state):
        #stateFeatures = self.getFeaturesFromState(state)
        return self.computeValueFromQValues(state)

    def saveTableToFile(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            #print "Saving QValueCounter of size: ", len(self.QValueCounter)
            pickle.dump(self.QValueCounter, output, pickle.HIGHEST_PROTOCOL)

    def loadTableFromFile(self, filename):
        with open(filename, 'rb') as input:
            self.QValueCounter = pickle.load(input)
            print "Size of loaded table: ", len(self.QValueCounter)

    def final(self, state):

        ReinforcementAgent.final(self, state)

        if self.episodesSoFar <= self.numTraining:
            self.saveTableToFile('featuresqtable.pkl')

    def getFeaturesFromState(self, state):
        #print "features: ", self.featExtractor.getFeatures(state, None)
        return self.featExtractor.getFeatures(state, None)

    def __del__(self):
        print "Qtable size: ", len(self.QValueCounter)


class PacmanFeatureQAgent(NewQLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        NewQLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        #print 'Starting getAction'
        action = NewQLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action



class PacmanNewFeatureQAgent(PacmanFeatureQAgent):

    def getFeaturesFromState(self, state):
        #print "features: ", self.featExtractor.getFeatures(state, None)

        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        selfPosition = state.getPacmanPosition()

        # new: take into account capsules and scared ghosts
        capsules = state.getCapsules()
        ghostStates = state.getGhostStates()

        nearestGhostPosition = self.closestGhost(selfPosition, ghosts, walls, ghostStates)
        nearestScaredGhostPosition = self.closestScaredGhost(selfPosition, ghosts, walls, ghostStates)
        nearestFoodPosition = self.closestFood(selfPosition, food, walls)
        nearestPillPosition = self.closestCapsule(selfPosition, capsules, walls)

        if nearestFoodPosition is None:
            foodDistance = Cuadrants.CENTER
        else:
            foodDistance = self.getQuadrant(selfPosition, nearestFoodPosition)

        if nearestGhostPosition is None:
            ghostDistance = Cuadrants.CENTER
        else:
            ghostDistance = self.getQuadrant(selfPosition, nearestGhostPosition)

        if nearestScaredGhostPosition is None:
            scaredGhostDistance = Cuadrants.CENTER
        else:
            scaredGhostDistance = self.getQuadrant(selfPosition, nearestScaredGhostPosition)

        if nearestPillPosition is None:
            capsuleDistance = Cuadrants.CENTER
        else:
            capsuleDistance = self.getQuadrant(selfPosition, nearestPillPosition)

        closest = self.getClosest(selfPosition, walls, food, ghosts, ghostStates)

        return (foodDistance, ghostDistance, scaredGhostDistance, capsuleDistance, closest)


    def distanceClosestFood(self, pos, food, walls):
        """
        closestFood -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
        return None


    def distanceClosestGhost(self, pos, ghosts, walls, ghostStates):
        """
        closestScaredGhost -- this is similar to the closestFood function, but for
        scared ghost.
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a scared ghost at this location then exit
            if any(ghost == (pos_x, pos_y) and ghostStates[ghosts.index(ghost)].scaredTimer <= 0 for ghost in ghosts):
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no scared ghost found
        return None


    def closestFood(self, pos, food, walls):
        """
        closestFood -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return pos_x, pos_y
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
        return None


    def closestGhost(self, pos, ghosts, walls, ghostStates):
        """
        closestScaredGhost -- this is similar to the closestFood function, but for
        scared ghost.
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a scared ghost at this location then exit
            if any(ghost == (pos_x, pos_y) and ghostStates[ghosts.index(ghost)].scaredTimer <= 0 for ghost in ghosts):
                return pos_x, pos_y
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no scared ghost found
        return None


    def closestScaredGhost(self, pos, ghosts, walls, ghostStates):
        """
        closestScaredGhost -- this is similar to the closestFood function, but for
        scared ghost.
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a scared ghost at this location then exit
            if any(ghost == (pos_x, pos_y) and ghostStates[ghosts.index(ghost)].scaredTimer > 0 for ghost in ghosts):
                return pos_x, pos_y
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no scared ghost found
        return None

    def closestCapsule(self, pos, capsules, walls):
        """
        closestCapsule -- this is similar to the closestFood function, but for
        scared ghost.
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a scared ghost at this location then exit
            if any(capsule == (pos_x, pos_y) for capsule in capsules):
                return pos_x, pos_y
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no scared ghost found
        return None


    def getQuadrant(self, pacmanPosition, position):

        if position == None:
            return None

        if position[0] == pacmanPosition[0]:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTH
            elif position[1] == pacmanPosition[1]:
                return Cuadrants.CENTER
            else:
                return Cuadrants.NORTH

        elif position[1] == pacmanPosition[1]:
            if position[0] < pacmanPosition[0]:
                return Cuadrants.EAST
            else:
                return Cuadrants.WEST

        elif position[0] < pacmanPosition[0]:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTHWEST
            else:
                return Cuadrants.NORTHWEST
        else:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTHEAST
            else:
                return Cuadrants.NORTHEAST

            # food closest: 0, ghost closest or same distance: 1

    def getClosest(self, pos, walls, food, ghosts, ghostStates):

        ghostDistance = self.distanceClosestGhost(pos, ghosts, walls, ghostStates)

        if ghostDistance is None:
            ghostDistance = 1000

        if ghostDistance <= 1:
            return 1
        else:
            return 0