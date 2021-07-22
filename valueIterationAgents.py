# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
        	newQvalues = self.values.copy()
        	for state in self.mdp.getStates():
        		listofQvals = []
        		if not self.mdp.isTerminal(state):
        			for action in self.mdp.getPossibleActions(state):
        				listofQvals.append(self.computeQValueFromValues(state, action))
        			bestValue = max(listofQvals)
        			newQvalues[state] = bestValue
        	self.values = newQvalues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q_val = 0
        transitionStatesandProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in transitionStatesandProbs:
        	nextStateReward = self.mdp.getReward(state, action, nextState)
        	Q_val += prob*(nextStateReward + self.discount*self.values[nextState])
        return Q_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        policies = util.Counter()
        for action in self.mdp.getPossibleActions(state):
        	policies[action] = self.computeQValueFromValues(state, action)
        return policies.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        numStates = len(allStates)
        for i in range(self.iterations):
        	thisState = allStates[i % numStates]
        	listofQvals = []
        	if self.mdp.isTerminal(thisState):
        		continue
        	for action in self.mdp.getPossibleActions(thisState):
        		listofQvals.append(self.computeQValueFromValues(thisState, action))
        	bestValue = max(listofQvals)
        	self.values[thisState] = bestValue



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        predecessorBoys = {}
        for state in allStates:
        	if not self.mdp.isTerminal(state):
        		possibleActions = self.mdp.getPossibleActions(state)
        		for action in possibleActions:
        			transitionStatesandProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        			for nextState, prob in transitionStatesandProbs:
        				if nextState in predecessorBoys:
        					predecessorBoys[nextState].add(state)
        				else: 
        					predecessorBoys[nextState] = {state}

        PriorityGuy = util.PriorityQueue()

        for state in allStates:
        	if not self.mdp.isTerminal(state):
        		listofQvals = []
        		currentVal = self.values[state]
        		for action in self.mdp.getPossibleActions(state):
        			listofQvals.append(self.computeQValueFromValues(state, action))
        		bestValue = max(listofQvals)
        		diff = abs(bestValue - currentVal)
        		PriorityGuy.update(state, -diff)

        for i in range(self.iterations):
        	if PriorityGuy.isEmpty():
        		break;
        	lastMan = PriorityGuy.pop()
        	if not self.mdp.isTerminal(lastMan):
        		listofQvals2 = []
        		for action in self.mdp.getPossibleActions(lastMan):
        			listofQvals2.append(self.computeQValueFromValues(lastMan, action))
        		bestValue = max(listofQvals2)
        		self.values[lastMan] = bestValue

        	for predecessor in predecessorBoys[lastMan]:
        		currValP = self.values[predecessor]
        		listofQvals = []
        		for action in self.mdp.getPossibleActions(predecessor):
        			listofQvals.append(self.computeQValueFromValues(predecessor, action))
        		bestValue = max(listofQvals)
        		diff = abs(bestValue - currValP)

        		if diff > self.theta:
        			PriorityGuy.update(predecessor, -diff)
















