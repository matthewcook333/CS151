# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # check ghosts, if we get too close to ghost, don't make the move
        for ghost in newGhostStates:
          ghostPos = ghost.getPosition()
          distance = abs(ghostPos[0] - newPos[0]) + abs(ghostPos[1] - newPos[1]) 
          if distance < 2:
            return 0        

        # check distance to closest food and how much food remaining
        minFoodDist = float('inf')
        foodCount = 0
        for y in range(newFood.height):
          for x in range(newFood.width):
            if newFood[x][y]:
              minFoodDist = min(minFoodDist, abs(newPos[0] - x) + abs(newPos[1] - y))
              foodCount += 1

        if foodCount == 0:
          return float('inf')

        # scale so that eating food is better than being close
        foodScore = (newFood.height*newFood.width) * ((newFood.height*newFood.width)-foodCount)
        # make sure that closer is better than farther from food
        foodDistScore = (newFood.height+newFood.width) - minFoodDist
        successorGameState.data.score = foodDistScore + foodScore
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        (score, action) = self.evaluateState(gameState, 0, self.depth)

        return action

    def evaluateState(self, state, agent, depth):
        # if at depth 0, evaluate the current state and return
        if depth == 0:
            return (self.evaluationFunction(state), "")

        actions = state.getLegalActions(agent)

        # if this action results in winning or losing, end the search down and
        # evaluate this terminal state
        if len(actions) == 0:
            return (self.evaluationFunction(state), "")

        # used to score the actions as we evaluate them
        maxScore = -float("inf")
        pacmanAction = ""
        minScore = float("inf")

        for action in actions:
            # find new state, next agent, and new depth and recurse on them
            newState = state.generateSuccessor(agent, action)
            newAgent = (agent + 1) % state.getNumAgents()
            newDepth = depth - 1 if newAgent == 0 else depth
            (score, subAction) = self.evaluateState(newState, newAgent, newDepth)

            if agent == 0:
                # pacman behavior (maximizer)
                if score > maxScore:
                    maxScore = score
                    pacmanAction = action
            else:
                # ghost adversary (minimizer)
                minScore = min(minScore, score)

        # if pacman, return max score. else return min score
        return (maxScore, pacmanAction) if agent == 0 else (minScore, "")

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        (score, action) = self.evaluateState(gameState, 0, self.depth, -float('inf'), float('inf'))

        return action


    def evaluateState(self, state, agent, depth, alpha, beta):
        # if at depth 0, evaluate the current state and return
        if depth == 0:
            return (self.evaluationFunction(state), "")

        actions = state.getLegalActions(agent)

        # if this action results in winning or losing, end the search down and
        # evaluate this terminal state
        if len(actions) == 0:
            return (self.evaluationFunction(state), "")

        # used to score the actions as we evaluate them
        maxScore = -float("inf")
        pacmanAction = ""
        minScore = float("inf")

        for action in actions:
            # find new state, next agent, and new depth and recurse on them
            newState = state.generateSuccessor(agent, action)
            newAgent = (agent + 1) % state.getNumAgents()
            newDepth = depth - 1 if newAgent == 0 else depth
            (score, subAction) = self.evaluateState(newState, newAgent, newDepth, alpha, beta)

            if agent == 0:
                # pacman behavior (maximizer)
                if score > maxScore:
                    maxScore = score
                    pacmanAction = action
                # pruning on beta, return current action
                if score > beta:
                    return (score, "")
                alpha = max(alpha, score)
            else:
                # ghost adversary (minimizer)
                minScore = min(minScore, score)
                # pruning on alpha
                if score < alpha:
                    return (score, "")
                beta = min(beta, score)

        # if pacman, return max score. else return min score
        return (maxScore, pacmanAction) if agent == 0 else (minScore, "")

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        (score, action) = self.evaluateState(gameState, 0, self.depth)

        return action

    def evaluateState(self, state, agent, depth):
        # if at depth 0, evaluate the current state and return
        if depth == 0:
            return (self.evaluationFunction(state), "")

        actions = state.getLegalActions(agent)

        # if this action results in winning or losing, end the search down and
        # evaluate this terminal state
        if len(actions) == 0:
            return (self.evaluationFunction(state), "")

        # used to score the actions as we evaluate them
        maxScore = -float("inf")
        pacmanAction = ""
        expectedScore = 0.0

        for action in actions:
            # find new state, next agent, and new depth and recurse on them
            newState = state.generateSuccessor(agent, action)
            newAgent = (agent + 1) % state.getNumAgents()
            newDepth = depth - 1 if newAgent == 0 else depth
            (score, subAction) = self.evaluateState(newState, newAgent, newDepth)

            if agent == 0:
                # pacman behavior (maximizer)
                if score > maxScore:
                    maxScore = score
                    pacmanAction = action
            else:
                # ghost adversary (added to expected score)
                expectedScore += score

        expectedScore = expectedScore/len(actions)
        # if pacman, return max score. else return expected score
        return (maxScore, pacmanAction) if agent == 0 else (expectedScore, "")



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # check ghosts, if we get too close to ghost, don't make the move
    for ghost in newGhostStates:
        ghostPos = ghost.getPosition()
        distance = abs(ghostPos[0] - newPos[0]) + abs(ghostPos[1] - newPos[1]) 
        if distance < 3:
            return 0        

    # check distance to closest food and how much food remaining
    minFoodDist = float('inf')
    foodCount = 0
    totalDist = 0
    for y in range(newFood.height):
        for x in range(newFood.width):
          if newFood[x][y]:
            foodDist = abs(newPos[0] - x) + abs(newPos[1] - y)
            minFoodDist = min(minFoodDist, foodDist)
            foodCount += 1
            totalDist += foodDist

    if foodCount == 0:
        return float('inf')

    # scale so that eating food is better than being close
    foodScore = (newFood.height*newFood.width) * ((newFood.height*newFood.width)-foodCount)
    # make sure that closer is better than farther from food
    foodDistScore = (newFood.height+newFood.width) - minFoodDist
    #print "food score: " + str(foodScore) + ", foodDistScore: " + str(foodDistScore)
    #currentGameState.data.score = foodDistScore + foodScore
    #return currentGameState.getScore()
    #print "state score: " + str(currentGameState.getScore()) + ", score: " + str(foodDistScore + foodScore)
    return foodDistScore + foodScore + currentGameState.getScore() - totalDist

# Abbreviation
better = betterEvaluationFunction

