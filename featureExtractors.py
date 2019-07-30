# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"Feature extractors for Pacman game states"

from game import Directions, Actions
import util


class FeatureExtractor:  
  def getFeatures(self, state, action):    
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 fors
      indicator functions.  
    """
    util.raiseNotDefined()

  def getFeatures(self, state):    
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.  
    """
    util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[(state,action)] = 1.0
    return feats

def closestFood(pos, food, walls):
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
      fringe.append((nbr_x, nbr_y, dist+1))
  # no food found
  return None

def closestFoodPos(pos, food, walls):
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
      return (pos_x, pos_y)
    # otherwise spread out from the location to its neighbours
    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    for nbr_x, nbr_y in nbrs:
      fringe.append((nbr_x, nbr_y, dist+1))
  # no food found
  return (0,0)


class SimpleExtractor(FeatureExtractor):
  """
  Returns simple features for a basic reflex Pacman:
  - whether food will be eaten
  - how far away the next food is
  - whether a ghost collision is imminent
  - whether a ghost is one step away
  """
  
  def getFeatures(self, state, action):
    # extract the grid of food and wall locations and get the ghost locations
    food = state.getFood()
    walls = state.getWalls()
    ghosts = state.getGhostPositions()

    features = util.Counter()
    
    features["bias"] = 1.0
    
    # compute the location of pacman after he takes the action
    x, y = state.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    
    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0
    
    dist = closestFood((next_x, next_y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = float(dist) / (walls.width * walls.height) 
    dist = closestFood((x, y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = float(dist) / (walls.width * walls.height) 
    features.divideAll(10.0)
    return features


## IA UC3M 2016
## You should understand the implementation of this class
class StateExtractor(FeatureExtractor):
  """
  Returns simple features for a basic reflex Pacman:
  - pacman position 
  - distance to the closest food from the pacman position
  - distance to the closest ghost from the pacman position
  """
  def isTerminalFeatures (self, featuresTuple):
    # No pacman (lose) or no food (win)
 
    # tuple example: ('IncGhostX': 4, 'IncGhostY': 0, 'posX': 5, 'posY': 1, 'FoodY': 4, 'FoodX': -3}
    # Condition: features["posX"]== None or features["#Food"] == 0
    return featuresTuple[2]  == None or featuresTuple[4] == None
    
## It returns the following features: pac-man position ("posX", "posY"), distance to the closest food ("FoodX", "FoodY") and distance to the closest ghost ("IncGhostX", "IncGhostY")
  def getFeatures(self, state):
    # extract the grid of food and wall locations and get the ghost locations
    food = state.getFood()
    walls = state.getWalls()
    ghosts = state.getGhostPositions()
    
    features = util.Counter()
    
   
    # compute the location of pacman
    pacmanPosition = state.getPacmanPosition()
    if pacmanPosition != None:
      x = pacmanPosition[0]
      y = pacmanPosition[1]
      features["posX"] = x
      features["posY"] = y
    else:
      features["posX"] = None
      features["posY"] = None


    # closest ghost
    if (state.getNumAgents() > 1):
      distancesToPacman = [(pos, util.manhattanDistance( pos, pacmanPosition )) for pos in ghosts]
      closestGhost_x, closestGhost_y  = min(distancesToPacman, key=lambda x: x[1])[0]
      features["IncGhostX"] = int (x - closestGhost_x)
      features["IncGhostY"] = int (y - closestGhost_y)

    # closest food
    if  state.getNumFood() != 0:
      (food_x, food_y) = closestFoodPos((x, y), food, walls)
      features["FoodX"] =  int (x- food_x)    
      features["FoodY"] =  int (y- food_y)
#      features["#Food"] =  True
#      features["#Food"] =  state.getNumFood()
    else:  
      features["FoodX"] =  None
      features["FoodY"] =  None
#      features["#Food"] =  False
#      features["#Food"] =  0


    return features
