# valueIterationAgents.py
# -----------------------
##
import mdp, util
import sys

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp = None, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbabilities(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        if (self.mdp != None):
            self.doValueIteration()


    def doValueIteration (self):
        # Write value iteration code here

        print "Iterations: ", self.iterations
        print "Discount: ", self.discount
        states = self.mdp.getStates()
        maxDelta = float("-inf")


        ##util.raiseNotDefined()
        #"*** YOUR CODE STARTS HERE ***"
        # Your code should include the implementation of value iteration
        # At the end it should show in the terminal the number of states considered in self.values and
        # the Delta between the last two iterations
       
        # Code to remove --- from here
        for i in range(self.iterations):
          newValues = util.Counter()
          for state in states:
            best = float("-inf")
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
              transitions = self.mdp.getTransitionStatesAndProbabilities(state, action)
              sumNextStates = 0
              for (nextState, probability) in transitions:
                reward = self.mdp.getReward(state, action, nextState)
                sumNextStates += probability*(reward + self.discount*self.values[nextState])
              best = max(best, sumNextStates)
            if best != float("-inf"):
              newValues[state] = best
          for state in states:
            if i == self.iterations-1:
            # last iteration
                maxDelta = max(maxDelta, abs(newValues[state] - self.values[state]))
            self.values[state] = newValues[state]
        print "Max Delta: ", maxDelta
        print "Number of states: ", len(self.values.keys())

        # Code to remove --- to here
        #"*** YOUR CODE FINISHES HERE ***"
        
    def setMdp( self, mdp):
        """
          Set an mdp.
        """
        self.mdp = mdp
        self.doValueIteration()

    def setDiscount( self, discount):
        """
          Set a discount
        """
        self.discount = discount

    def setIterations( self, iterations):
        """
          Set a number of iterations
        """
        self.iterations = iterations
       
       
    def getValue(self, state):
        """
          Return the value of the state
        """
        return self.values[state]
        
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        ##util.raiseNotDefined()
        #"*** YOUR CODE STARTS HERE ***"
        # Code to remove --- from here
        transitions = self.mdp.getTransitionStatesAndProbabilities(state, action)
        qvalue = 0
        for (nextState, probability) in transitions:
          reward = self.mdp.getReward(state, action, nextState)
          qvalue += probability *(reward + self.discount*self.values[nextState])
        # Code to remove --- to here
        #"*** YOUR CODE FINISHES HERE ***"
        
        return qvalue
    

    def showPolicy( self ):

        """
          Print the policy
        """
        
        states = self.mdp.getStates()
        for state in states:
            print "Policy\n", state, self.getPolicy(state)


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        ##util.raiseNotDefined()
        #"*** YOUR CODE STARTS HERE ***"
       
        # Code to remove --- from here
        resultingAction = None
        if self.mdp.isTerminal(state):
            return resultingAction
        else:
            bestq = float("-inf")
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
              qvalue = self.computeQValueFromValues(state, action)
              if qvalue > bestq:
                bestq = qvalue
                resultingAction = action
            return resultingAction

        # Code to remove --- to here
        #"*** YOUR CODE FINISHES HERE ***"

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getPolicy(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getAction(state)

    
    def getQValue(self, state, action):
        "Returns the Q value."        
        return self.computeQValueFromValues(state, action)

    def getPartialPolicy(self, stateL):
        "Returns the partial policy at the state. Random for unkown states"        
        state = self.mdp.stateToHigh(stateL)
        if self.mdp.isKnownState(state):
            return self.computeActionFromValues(state)
        else:
            # random action
            return util.random.choice(stateL.getLegalActions()) 

