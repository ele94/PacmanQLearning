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
from numpy import zeros
import numpy

import random,util,math,pickle


class QLearningAgent(ReinforcementAgent):
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
    
    qtable = ""
    num_states = 25
    num_actions = 4
    qtable_file = 'qtable.npy'
    QValueCounter = ""
    
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        "*** YOUR CODE HERE ***"
        # "inicializamos las variables q (aqui podriamos hacer que los valores q se pasen como parametro)" 
        # self.alpha = 0.5
        # self.discount = 0.8
        # self.epsilon = 0.8
        
        "Tambien leemos la q-table"
        self.qtable = self.readQTable()
        
        "Otra qtable"
        self.QValueCounter = util.Counter()

        # cargamos qvalues ya creados (?)
        self.loadQTableFromFile('tables/smallgridfourthagent.pkl')
        print "Loading qtable"

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        ReinforcementAgent.final(self, state)


        if self.episodesSoFar <= self.numTraining:
            self.saveQTableToFile('qvalues.pkl')
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

        
    def __del__(self):
        # # actualizamos la q table
        # # registramos el estado actual
        # self.current_state = self.getQStateIndex(gameState)
        # self.current_score = gameState.getScore()
        # self.reward = self.current_score - self.previous_score 
        
        # # self.experiencia = [self.previous_state, self.previous_move, self.current_state, reward]
        # # update qtable
        # if(self.previous_move < 5):
            # self.updateQTable(self.previous_state, self.previous_move, self.current_state, self.reward)

        print "QTable size: ", len(self.QValueCounter)
        #self.saveQTable()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qtable[self.getStateIndex(state)][self.getActionIndex(action)]
        # util.raiseNotDefined()
        
        # return self.QValueCounter[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return None

        best_action = None; best_value = float('-inf')
        for action in self.getLegalActions(state):
            if action != 'Stop' and self.getQValue(state, action) > best_value:
                best_value = self.getQValue(state, action)
                best_action = action
        return best_action
        
        # CAMBIAR LUEGO
        # legalActions = self.getLegalActions(state)
        # return random.choice(legalActions)

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
        if 'Stop' in legalActions: legalActions.remove("Stop")
        action = None
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return None # Terminal State, return None
        # politica greedy
        random_number = random.uniform(0, 1)
        if(random_number > self.epsilon):
            action = self.getPolicy(state)
        else:
            action = random.choice(legalActions) # Explore
        
        # al parecer no hace falta hacer el update porque se hace solo??????
        # # actualizamos la tabla q
        # nextState = state.generatePacmanSuccessor(action)
        # print "nextState: ", nextState
        # print "state: ", state
        # reward = self.getReward(state, nextState)
        # self.update(state, action, nextState, reward)
        
        # util.raiseNotDefined()
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
        
        stateIndex = self.getStateIndex(state)
        nextStateIndex = self.getStateIndex(nextState)
        actionIndex = self.getActionIndex(action)
        
        self.qtable[stateIndex][actionIndex] = self.qtable[stateIndex][actionIndex] + (self.alpha * ( reward + 
            (self.discount * max(self.qtable[nextStateIndex])) - self.qtable[stateIndex][actionIndex]))
        
        
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
        
        
        
        
    """
        A partir de aqui se definen las funciones creadas para anhadir la funcionalidad
    """
    
    def getQState(self, state):
        return 0
    
    def readQTable(self):
        import os
        exists = os.path.isfile(self.qtable_file)
        if exists:
            t = numpy.load(self.qtable_file)
        else:
            t = zeros([self.num_states,self.num_actions])
        return t
            
    def saveQTable(self):
        numpy.save(self.qtable_file, self.qtable)
        
        
    def getActionIndex(self, action):
        return 0
    
    def getStateIndex(self, state):
        
        # nearestGhostPosition = state.getPositionNearestGhost()
        # nearestFoodPosition = state.getPositionNearestFood()
        # selfPosition = state.getPacmanPosition()
        
        # xFoodDistance = selfPosition[0] - nearestFoodPosition[0]
        # yFoodDistance = selfPosition[1] - nearestFoodPosition[1]
        
        # xGhostDistance = selfPosition[0] - nearestGhostPosition[0]
        # yGhostDistance = selfPosition[1] - nearestGhostPosition[1]
    
        return 0
              
    def saveQTableToFile(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.QValueCounter, output, pickle.HIGHEST_PROTOCOL)

    def loadQTableFromFile(self, filename):
        with open(filename, 'rb') as input:
            self.QValueCounter = pickle.load(input)
    
    def getNearestGhostPosition(self, state):
        pacmanPosition = state.getPacmanPosition()
        closestGhostDistance = 1000
        closestGhostPosition = None
    
        for position in state.getGhostPositions():
            distance = util.manhattanDistance(pacmanPosition, position)
            if distance < closestGhostDistance:
                closestGhostDistance = distance
                closestGhostPosition = position
                
        return closestGhostPosition
            
    
    def getNearestFoodPosition(self, state):
        pacmanPosition = state.getPacmanPosition()
        closestFoodDistance = 1000
        closestFoodPosition = None
    
        for position in state.getFoodPositions():
            distance = util.manhattanDistance(pacmanPosition, position)
            if distance < closestFoodDistance:
                closestFoodDistance = distance
                closestFoodPosition = position
                
        return closestFoodPosition
              

class PacmanQAgent(QLearningAgent):
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
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
        
        
class PacmanFirstQAgent(PacmanQAgent):

    def __init__(self, **args):
        self.index = 0  # This is always Pacman
        PacmanQAgent.__init__(self, **args)
    
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
       
        best_action = self.computeActionFromQValues(nextState)
        
        self.QValueCounter[(self.getQState(state), action)] = ((1-self.alpha)*self.getQValue(state, action) +
                                               self.alpha*(reward+self.discount*self.getQValue(nextState, best_action)))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QValueCounter[(self.getQState(state), action)]   

    def getQState(self, state):

        nearestGhostPosition = self.getNearestGhostPosition(state)
        nearestFoodPosition = self.getNearestFoodPosition(state)
        selfPosition = state.getPacmanPosition()
        
        if nearestFoodPosition is None:
            xFoodDistance = 0
            yFoodDistance = 0
        else:
            xFoodDistance = selfPosition[0] - nearestFoodPosition[0]
            yFoodDistance = selfPosition[1] - nearestFoodPosition[1]
        
        if nearestGhostPosition is None:
            xGhostDistance = 0
            yGhostDistance = 0
        else:
            xGhostDistance = selfPosition[0] - nearestGhostPosition[0]
            yGhostDistance = selfPosition[1] - nearestGhostPosition[1]
        
        return (xFoodDistance, yFoodDistance, xGhostDistance, yGhostDistance)
       

    
    # def readQTable(self):
        # import os
        # exists = os.path.isfile(self.qtable_file)
        # if exists:
            # t = numpy.load(self.qtable_file)
        # else:
            # t = zeros([self.num_states,self.num_actions])
        # return t
            
    # def saveQTable(self):
        # numpy.save(self.qtable_file, self.qtable)

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


class PacmanSecondQAgent(PacmanQAgent):

    def __init__(self, **args):
        self.index = 0  # This is always Pacman
        PacmanQAgent.__init__(self, **args)
    
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
       
        best_action = self.computeActionFromQValues(nextState)
        
        self.QValueCounter[(self.getQState(state), action)] = ((1-self.alpha)*self.getQValue(state, action) +
                                               self.alpha*(reward+self.discount*self.getQValue(nextState, best_action)))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QValueCounter[(self.getQState(state), action)]   

    def getQState(self, state):

        nearestGhostPosition = self.getNearestGhostPosition(state)
        nearestFoodPosition = self.getNearestFoodPosition(state)
        selfPosition = state.getPacmanPosition()
        
        if nearestFoodPosition is None:
            foodDistance = Cuadrants.CENTER
        else:
            foodDistance = self.getQuadrant(selfPosition, nearestFoodPosition)
        
        if nearestGhostPosition is None:
            ghostDistance = Cuadrants.CENTER
        else:
            ghostDistance = self.getQuadrant(selfPosition, nearestGhostPosition)
        
        return (foodDistance, ghostDistance)
        
    def getQuadrant(self, pacmanPosition, position):
        
        if position is None:
            return None
            
        if position[0] == pacmanPosition[0]:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTH
            elif position[1] == pacmanPosition[1]:
                return Cuadrants.CENTER
            else: return Cuadrants.NORTH
        
        elif position[1] == pacmanPosition[1]:
            if position[0] < pacmanPosition[0]:
                return Cuadrants.EAST
            else: return Cuadrants.WEST
            
        elif position[0] < pacmanPosition[0]:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTHWEST
            else: return Cuadrants.NORTHWEST
        else:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTHEAST
            else: return Cuadrants.NORTHEAST
  

class PacmanThirdQAgent(PacmanQAgent):
  
    def __init__(self, **args):
        self.index = 0  # This is always Pacman
        PacmanQAgent.__init__(self, **args)
    
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
       
        best_action = self.computeActionFromQValues(nextState)
        
        self.QValueCounter[(self.getQState(state), action)] = ((1-self.alpha)*self.getQValue(state, action) +
                                               self.alpha*(reward+self.discount*self.getQValue(nextState, best_action)))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QValueCounter[(self.getQState(state), action)]   

    def getQState(self, state):

        nearestGhostPosition = self.getNearestGhostPosition(state)
        nearestFoodPosition = self.getNearestFoodPosition(state)
        selfPosition = state.getPacmanPosition()
        
        wallTypeFood = None
        wallTypeGhost = None
        
        if nearestFoodPosition is None:
            foodDistance = Cuadrants.CENTER
        else:
            foodDistance = self.getQuadrant(selfPosition, nearestFoodPosition)
            wallTypeFood = self.checkMuro(state, selfPosition[0], selfPosition[1], nearestFoodPosition[0], nearestFoodPosition[1])

        
        if nearestGhostPosition is None:
            ghostDistance = Cuadrants.CENTER
        else:
            ghostDistance = self.getQuadrant(selfPosition, nearestGhostPosition)
            wallTypeGhost =  self.checkMuro(state, selfPosition[0], selfPosition[1], nearestGhostPosition[0], nearestGhostPosition[1])       
        
        return foodDistance, ghostDistance, wallTypeFood, wallTypeGhost
        
    def getQuadrant(self, pacmanPosition, position):
        
        if position is None:
            return None
            
        if position[0] == pacmanPosition[0]:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTH
            elif position[1] == pacmanPosition[1]:
                return Cuadrants.CENTER
            else: return Cuadrants.NORTH
        
        elif position[1] == pacmanPosition[1]:
            if position[0] < pacmanPosition[0]:
                return Cuadrants.EAST
            else: return Cuadrants.WEST
            
        elif position[0] < pacmanPosition[0]:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTHWEST
            else: return Cuadrants.NORTHWEST
        else:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTHEAST
            else: return Cuadrants.NORTHEAST 
  
  
    #Metodo que revisa si hay un muro entre Pacman y otro elemento cuyas posiciones son introducidas como argumentos
    def checkMuro(self,state, colpac, filapac, colfant, filafant):
        walls= state.data.layout.walls
        xmax=state.data.layout.width
        ymax=state.data.layout.height
        tipofila=0
        tipocol=0
        
        # Si Pacman esta encima del fantasma
        if filapac>filafant:
            fila=filapac
            while fila>filafant:
                fila -=1
                if walls[colpac][fila] and fila>0 :
                    tipofila= self.getTipo(fila,colpac,walls,xmax,ymax)
                    if tipocol==2 or tipocol==5 or tipocol==6: tipocol=1
                    return tipofila
        # Si esta debajo
        if filapac<filafant:
            fila=filapac
            while fila<filafant:
                fila +=1
                if walls[colpac][fila] and fila<ymax:
                    tipofila=self.getTipo(fila,colpac,walls,xmax,ymax)
                    if tipocol == 2 or tipocol == 5 or tipocol == 6: tipocol=1
                    return tipofila
        # A la derecha
        if colpac>colfant:
            col=colpac
            while col>colfant:
                col -=1
                if walls[col][filapac] and col>0:
                    tipocol=self.getTipo(filapac,col,walls,xmax,ymax)
                    if tipocol==1 or tipocol==3 or tipocol==4: tipocol=2
                 
                    return tipocol
        # A la izquierda
        if colpac<colfant:
            col=colpac
            while col<colfant:
                col +=1
                if walls[col][filapac]and col<xmax:
                    tipocol=self.getTipo(filapac,col,walls,xmax,ymax)
                    if tipocol==1 or tipocol==3 or tipocol==4: tipocol=2
                    return tipocol
              
        return 0
    
    # Devuelve el tipo de muro cuyas posiciones son introducidas como argumentos
    def getTipo(self,fila,col,walls,maxcol,maxfila):
        clasif=-1
        f=fila
        cerradoIz1=False
        cerradoDer=False
        cerradoabajo=False
        cerradoarriba=False
        c=col
        
        clasif,cerradoIz1,cerradoDer= self.horizontal(walls,fila,col,maxcol)
        if clasif == -1:
          clasif,cerradoIz1,cerradoDer= self.vertical(walls,fila,col,maxfila,maxcol)

        if clasif == 1 and cerradoIz1: clasif = 3
        if clasif == 1 and cerradoDer: clasif = 4
        if clasif == 2 and cerradoabajo: clasif = 5
        if clasif == 2 and cerradoarriba: clasif = 6
        return clasif

    # En caso de que el muro sea horizontal define si esta cerrado o no y por que lado
    def horizontal(self,walls,fila,col,maxcol):
        c=col
        clasif=-1
        cerradoIz1=False
        cerradoDer=False
        while c>0 and (walls[c-1][fila]):
           c-=1
           clasif =1
           if c==0:
               cerradoIz1=True
    
        c=col
        while  c<maxcol-1 and walls[c+1][fila]:
           clasif =1
           c+=1
           if c==maxcol-1:
               cerradoDer=True
        
        return clasif,cerradoIz1,cerradoDer

    # En caso de que el muro sea vertical define si esta cerrado o no y por que lado
    def vertical(self,walls,fila,col,maxfila,maxcol):
        f=fila
        clasif=-1
        clasif2=-1
        cerradoIz1=-1
        cerradoDer=-1
        cerradoabajo=False
        cerradoarriba=False
        while (walls[col][f-1]) and f>0:
           clasif =2
           f-=1
           if not walls[col][f-1] and f >0:
               clasif2,cerradoIz1,cerradoDer= self.horizontal(walls,f,col,maxcol)
               if cerradoIz1 : clasif=3
               if cerradoDer : clasif=4
           if f==0:
               cerradoabajo=True
       
        f=fila
        while (walls[col][f+1]) and f<maxfila-1:
           clasif=2
           f+=1
           if not walls[col][f+1] and f<maxfila:
              clasif2,cerradoIz1,cerradoDer= self.horizontal(walls,f,col,maxcol)
              if cerradoIz1 : clasif=3
              if cerradoDer : clasif=4
           if f==maxfila:
               cerradoarriba=True
       
        return clasif,cerradoabajo,cerradoarriba

    # Identifica el cuadrante del elemento introducido como argumento respecto de Pacman
    def cuadrante(self, colpac, filapac, colfant, filafant):
        cuad = -1
        if filafant > filapac:
            # Fantasma arriba-izq
            if colfant < colpac: cuad = 0
            # Fantasma arriba-derecha
            elif colfant > colpac: cuad = 2
            # Fantasma arriba
            else: cuad = 1
        elif filafant < filapac:
            # Fantasma abajo-izq
            if colfant < colpac: cuad = 6
            # Fantasma abajo-derecha
            elif colfant > colpac: cuad = 4
            # Fantasma abajo
            else: cuad = 5
        else:
            # Fantasma izquierda
            if colfant < colpac: cuad = 7
            # Fantasma derecha
            elif colfant > colpac: cuad = 3
            # Fantasma encima de Pacman
            else: cuad = 8
        return cuad


class PacmanFourthQAgent(PacmanQAgent):
  
    def __init__(self, **args):
        self.index = 0  # This is always Pacman
        PacmanQAgent.__init__(self, **args)
    
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
       
        best_action = self.computeActionFromQValues(nextState)
        
        self.QValueCounter[(self.getQState(state), action)] = ((1-self.alpha)*self.getQValue(state, action) +
                                               self.alpha*(reward+self.discount*self.getQValue(nextState, best_action)))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QValueCounter[(self.getQState(state), action)]   

    def getQState(self, state):

        nearestGhostPosition = self.getNearestGhostPosition(state)
        nearestFoodPosition = self.getNearestFoodPosition(state)
        selfPosition = state.getPacmanPosition()
        
        wallTypeFood = None
        wallTypeGhost = None
        
        if nearestFoodPosition is None:
            foodDistance = Cuadrants.CENTER
        else:
            foodDistance = self.getQuadrant(selfPosition, nearestFoodPosition)
            wallTypeFood = self.checkMuro(state, selfPosition[0], selfPosition[1], nearestFoodPosition[0], nearestFoodPosition[1])

        
        if nearestGhostPosition is None:
            ghostDistance = Cuadrants.CENTER
        else:
            ghostDistance = self.getQuadrant(selfPosition, nearestGhostPosition)
            wallTypeGhost =  self.checkMuro(state, selfPosition[0], selfPosition[1], nearestGhostPosition[0], nearestGhostPosition[1])

        closest = self.getClosest(selfPosition, nearestFoodPosition, nearestGhostPosition)
        
        return (foodDistance, ghostDistance, wallTypeFood, wallTypeGhost, closest)



    def getQuadrant(self, pacmanPosition, position):
        
        if position == None:
            return None
            
        if position[0] == pacmanPosition[0]:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTH
            elif position[1] == pacmanPosition[1]:
                return Cuadrants.CENTER
            else: return Cuadrants.NORTH
        
        elif position[1] == pacmanPosition[1]:
            if position[0] < pacmanPosition[0]:
                return Cuadrants.EAST
            else: return Cuadrants.WEST
            
        elif position[0] < pacmanPosition[0]:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTHWEST
            else: return Cuadrants.NORTHWEST
        else:
            if position[1] < pacmanPosition[1]:
                return Cuadrants.SOUTHEAST
            else: return Cuadrants.NORTHEAST 
  
    # food closest: 0, ghost closest or same distance: 1
    def getClosest(self, pacmanPosition, foodPosition, ghostPosition):
    
        if foodPosition is None:
            foodDistance = 1000
        else:
            foodDistance = util.manhattanDistance(pacmanPosition, foodPosition)
        if ghostPosition is None:
            ghostDistance = 1000
        else:
            ghostDistance = util.manhattanDistance(pacmanPosition, ghostPosition)
        
        if foodDistance < ghostDistance:
            return 0
        else: return 1
        
        
  
    #Metodo que revisa si hay un muro entre Pacman y otro elemento cuyas posiciones son introducidas como argumentos
    def checkMuro(self,state, colpac, filapac, colfant, filafant):
        walls= state.data.layout.walls
        xmax=state.data.layout.width
        ymax=state.data.layout.height
        tipofila=0
        tipocol=0
        
        #Si Pacman esta encima del fantasma
        if filapac>filafant:
            fila=filapac
            while fila>filafant:
                fila -=1
                if walls[colpac][fila] and fila>0 :
                 tipofila= self.getTipo(fila,colpac,walls,xmax,ymax)
                 if tipocol==2 or tipocol==5 or tipocol==6: tipocol=1
                 return tipofila
        #Si esta debajo        
        if filapac<filafant:
            fila=filapac
            while fila<filafant:
                fila +=1
                if walls[colpac][fila] and fila<ymax:
                 tipofila=self.getTipo(fila,colpac,walls,xmax,ymax)
                 if tipocol==2 or tipocol==5 or tipocol==6: tipocol=1
                 return tipofila
        #A la derecha
        if colpac>colfant:
            col=colpac
            while col>colfant:
                col -=1
                if walls[col][filapac] and col>0:
                 tipocol=self.getTipo(filapac,col,walls,xmax,ymax)
                 if tipocol==1 or tipocol==3 or tipocol==4: tipocol=2
                 
                 return tipocol
        #A la izquierda
        if colpac<colfant:
            col=colpac
            while col<colfant:
                col +=1
                if walls[col][filapac]and col<xmax:
                 tipocol=self.getTipo(filapac,col,walls,xmax,ymax)
                 if tipocol==1 or tipocol==3 or tipocol==4: tipocol=2
                 return tipocol
              
        return 0
    
    #Devuelve el tipo de muro cuyas posiciones son introducidas como argumentos
    def getTipo(self,fila,col,walls,maxcol,maxfila):
        clasif=-1
        f=fila
        cerradoIz1=False
        cerradoDer=False
        cerradoabajo=False
        cerradoarriba=False
        c=col
        
        clasif,cerradoIz1,cerradoDer= self.horizontal(walls,fila,col,maxcol)
        if clasif == -1:
          clasif,cerradoIz1,cerradoDer= self.vertical(walls,fila,col,maxfila,maxcol)
        
        
        
        if clasif==1 and cerradoIz1 : clasif=3
        if clasif==1 and cerradoDer : clasif=4
        if clasif==2 and cerradoabajo: clasif=5
        if clasif==2 and cerradoarriba: clasif=6
        return clasif
       

    #En caso de que el muro sea horizontal define si esta cerrado o no y por que lado   
    def horizontal(self,walls,fila,col,maxcol):
        c=col
        clasif=-1
        cerradoIz1=False
        cerradoDer=False
        while c>0 and (walls[c-1][fila]):
           c-=1
           clasif =1
           if c==0:
               cerradoIz1=True
    
        c=col
        while  c<maxcol-1 and walls[c+1][fila]:
           clasif =1
           c+=1
           if c==maxcol-1:
               cerradoDer=True
        
        return clasif,cerradoIz1,cerradoDer

    #En caso de que el muro sea vertical define si esta cerrado o no y por que lado   
    def vertical(self,walls,fila,col,maxfila,maxcol):
        f=fila
        clasif=-1
        clasif2=-1
        cerradoIz1=-1
        cerradoDer=-1
        cerradoabajo=False
        cerradoarriba=False
        while (walls[col][f-1]) and f>0:
           clasif =2
           f-=1
           if not walls[col][f-1] and f >0:
               clasif2,cerradoIz1,cerradoDer= self.horizontal(walls,f,col,maxcol)
               if cerradoIz1 : clasif=3
               if cerradoDer : clasif=4
           if f==0:
               cerradoabajo=True
       
        f=fila
        while (walls[col][f+1]) and f<maxfila-1:
           clasif=2
           f+=1
           if not walls[col][f+1] and f<maxfila:
              clasif2,cerradoIz1,cerradoDer= self.horizontal(walls,f,col,maxcol)
              if cerradoIz1 : clasif=3
              if cerradoDer : clasif=4
           if f==maxfila:
               cerradoarriba=True
       
        return clasif,cerradoabajo,cerradoarriba

    #Identifica el cuadrante del elemento introducido como argumento respecto de Pacman
    def cuadrante(self, colpac, filapac, colfant, filafant):
        cuad = -1
        if(filafant > filapac): 
            #Fantasma arriba-izq
            if(colfant < colpac): cuad = 0
            #Fantasma arriba-derecha
            elif(colfant > colpac): cuad = 2
            #Fantasma arriba
            else: cuad = 1
        elif(filafant < filapac):
            # Fantasma abajo-izq
            if(colfant < colpac): cuad = 6
            # Fantasma abajo-derecha
            elif(colfant > colpac): cuad = 4
            # Fantasma abajo
            else: cuad = 5
        else:
            #Fantasma izquierda
            if(colfant < colpac): cuad = 7
            #Fantasma derecha
            elif(colfant > colpac): cuad = 3
            #Fantasma encima de Pacman
            else: cuad = 8
        return cuad


""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" NUEVO AGENTE PARA LA NUEVA FUNCIONALIDAD APPROXIMATE "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
