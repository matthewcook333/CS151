# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    s = util.Stack()
    closed = set()
    
    s.push([(problem.getStartState(), 'INITIAL', 0)])

    while not s.isEmpty():
        currentSuccList = s.pop()
        currentState = currentSuccList[0][0]
        #print currentSuccList
        #print not s.isEmpty()
        if problem.isGoalState(currentState):
            return getActionListFromSuccList(currentSuccList)
        if currentState not in closed:
            closed.add(currentState)
            for succ in problem.getSuccessors(currentState):
                # visited = False
                # for (state, action, cost) in currentSuccList:
                #     print state + " " + succ[0]
                #     if state == succ[0]:
                #         print "V: " + state
                #         visited = True
                #         break
                #     else:
                #         print "N: " + state
                # if not visited:
                #     newPath = list(currentSuccList)
                #     newPath.insert(0, succ)
                #     s.push(newPath)
                newPath = list(currentSuccList)
                newPath.insert(0, succ)
                s.push(newPath)
    print "FAILURE"

def getActionListFromSuccList(succList):
    """
    Given a list of successors, extract the list of actions.
    """
    actionList = []
    for succ in succList[:len(succList)-1]:
        actionList.insert(0, succ[1])
    return actionList

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    s = util.Queue()
    closed = set() 
    s.push([(problem.getStartState(), 'INITIAL', 0)])

    while not s.isEmpty():
        currentSuccList = s.pop()
        currentState = currentSuccList[0][0]
        if problem.isGoalState(currentState):
            return getActionListFromSuccList(currentSuccList)
        if currentState not in closed:
            closed.add(currentState)
            for succ in problem.getSuccessors(currentState):
                newPath = list(currentSuccList)
                newPath.insert(0, succ)
                s.push(newPath)
    print "FAILURE"

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    s = util.PriorityQueue()
    closed = set() 
    s.push(([(problem.getStartState(), 'INITIAL', 0)], 0), 0)
    
    while not s.isEmpty():
        currentPath = s.pop()
        currentState = currentPath[0][0][0]
        currentCost = currentPath[1]
        if problem.isGoalState(currentState):
            return getActionListFromSuccList(currentPath[0])
        if currentState not in closed:
            closed.add(currentState)
            for succ in problem.getSuccessors(currentState):
                newPath = list(currentPath[0])
                newPath.insert(0, succ)
                newCost = currentCost + succ[2]
                s.push((newPath, newCost), newCost)
    print "FAILURE"

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
