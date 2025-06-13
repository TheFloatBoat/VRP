#CSV or XLSX format: NodeNumber (MUST be unique for every single node), NodeType(e.g. 0 for depot 1 for customer), X, Y, Demand
#Use Euclidean distances for now


##VRP Solver!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print("VRP Solver!")
dataframe = pd.read_csv("C:/Users/User/VRPNodes115.csv")




nodeCount = len(dataframe.values)
xPoints = dataframe["X"].values
yPoints = dataframe["Y"].values
vehicleCapacity = int(input("Input Vehicular Capacity: "))




distances = np.zeros((nodeCount, nodeCount)) #Making distance matrix
for i in range(nodeCount):
    for j in range (nodeCount):
        euclideanD = ((xPoints[j] - xPoints[i])**2 + (yPoints[j] - yPoints[i])**2)**0.5
        distances[i][j] = euclideanD
        distances[j][i] = euclideanD


startPoint = 0
depots = dataframe.loc[dataframe["NodeType"]==0]
totalRouteDist = 0 #Lvl 1 metric, from lvl 3 best
totalMinimumDist = None
bestRoute = None #Lvl 1 best
decidedRoute = [] #Lvl 1 competitor
decidedRouteWithBackTrips = []
minNodeCode = None




#Lvl 2 of iteration is a repeat-purposed one and does not compare anything for selection, simply has a purpose of executing lvl 3 iterations an appropriate number of times on different items




def chooseNextNode(currentLoc, cannotServeLocal):
  #takes in the current location and the things you cant serve, edits demandOfMinNode, minNodeCode, and minimumDist


  #Nodes in cannotServe have demands which exceed vehicle capacity left
    global demandOfMinNode, minNodeCode, minimumDist    #minNodeCode is the node number of the minimum node


    for index2, row2 in dataframe.iterrows():  #This is a comparison-purposed iteration
      if row2["NodeType"] != 0 and row2["NodeNumber"] != currentLoc and row2["NodeNumber"] not in decidedRoute: #Do not consider this node if it is a depot, the current node or a node that has already been visited
          if (row2["Demand"] > vehicleCapacityLeft).any():
              cannotServeLocal.append(row2["NodeNumber"])
              continue


          #it would be more fun if they were one giant if else but this is technically better cause you save a lookup when it reaches the end of an "arm" but ughhh
          distanceToSuggested = distances[currentLoc][index2] #Lvl 3 competitor   #suggested refers to the suggestion of the loop not the best


          #yes its weird to check this every single iteration but its a small check and if you initialise first you gotta make sure its still valid and the performance gains cancel out
          if minimumDist == None: #If this is the first iteration
              minimumDist = distanceToSuggested
              minNodeCode = index2
              demandOfMinNode = row2["Demand"]


          elif minimumDist > distanceToSuggested: #distanceToSuggested beats previous minimumDist, becomes new minimumDist.
              minimumDist = distanceToSuggested
              minNodeCode = index2
              demandOfMinNode = row2["Demand"]
          else:
              pass #aka the suggestion is longer


def chooseReturnDepot(currentLoc):
    global vehicleCapacity, vehicleCapacityLeft, totalRouteDist, depots, decidedRouteWithBackTrips, distances
    minDepotDist = None
    for depotIndex, depotRow in depots.iterrows():
        if minDepotDist is None:
            minDepotDist = distances[currentLoc][int(depotRow["NodeNumber"])]
            closestDepot = int(depotRow["NodeNumber"])
        elif minDepotDist > distances[currentLoc][int(depotRow["NodeNumber"])]:
            closestDepot = int(depotRow["NodeNumber"])
            minDepotDist = distances[currentLoc][int(depotRow["NodeNumber"])]
    totalRouteDist += distances[currentLoc][closestDepot]
    decidedRouteWithBackTrips.append(closestDepot)
    vehicleCapacityLeft = vehicleCapacity


def calculate2DDistances(backtripRoute):
    global distances #sorry thats yet another global variable ;(
    backtripRouteNaked = []
    for i in range(len(backtripRoute)):
        for j in range(len(backtripRoute[i])): #Address 2-dimensionality
            backtripRouteNaked.append(backtripRoute[i][j])
    previousNode = None
    itercount = -1
    for node in backtripRouteNaked: #1-dimension
        itercount +=1
        if previousNode == None:
            previousNode = node
            continue
        elif previousNode != None and previousNode == node:
            del backtripRouteNaked[itercount]
            itercount -= 1
    itercount = -1
    backtripRouteDist = 0
    for node in backtripRouteNaked:
        itercount +=1
        try:
            backtripRouteDist += distances[node][backtripRouteNaked[itercount+1]]
        except:
            continue
    return backtripRouteDist






#CalculatePath functions now need to be double iterations because minimum distance is not just from currentlyAt to depot, but also then towards next node.
def calculatePath(currentLoc, routeList, routeListOnlyCustomer): #CurrentLocation -- depot -- nextNode1 -- nextNode2. nextNode2 to be NN of nextNode1, which is NN of currentLocation
    global vehicleCapacity, vehicleCapacityLeft, totalRouteDist, depots, decidedRouteWithBackTrips, distances, path1Dist, vehicleCapacity, vehicleCapacityLeft, minNodeCode
    minPathDist = None


    #finds the best depot to node combo
    for depotIndex, depotRow in depots.iterrows(): #minimum of first 3 steps
        for nextNodeIndex, nextNodeRow in dataframe.iterrows():
            distToDepotAndNodeAfter = distances[depotRow["NodeNumber"]][currentLoc] + distances[depotRow["NodeNumber"]][nextNodeRow["NodeNumber"]]
            if minPathDist is None:
                minPathDist = distToDepotAndNodeAfter
                next11 = depotRow["NodeNumber"]  #next node 1 for path 1
                next21 = nextNodeRow["NodeNumber"]  #next node 2 for path 1
            elif distToDepotAndNodeAfter < minPathDist:
                minPathDist = distToDepotAndNodeAfter
                next11 = depotRow["NodeNumber"] #next node 1 for path 1
                next21 = nextNodeRow["NodeNumber"]  #next node 2 for path 1


    vehicleCapacityLeftIfNotPath1 = vehicleCapacityLeft
    vehicleCapacityLeft = vehicleCapacity #gives max vehicle capacity for iteration since would visit depot first if takes path 1
    cannotServe = []
    chooseNextNode(currentLoc, cannotServe)
    path1Dist = minPathDist + minimumDist
    next31 = minNodeCode #next31 represents the third next node the vehicle would travel to if it takes path 1.
    minPathDist = None
    vehicleCapacityLeft = vehicleCapacityLeftIfNotPath1 #sets vehicleCapacityLeft to original since would not visit depot yet if takes path 2
    cannotServe = []
    chooseNextNode(currentLoc, cannotServe)
    next12 = minNodeCode


    for depotIndex, depotRow in depots.iterrows():
        for nextNodeIndex, nextNodeRow in dataframe.iterrows():
            if minPathDist is None:
                minPathDist = distances[next12][depotRow["NodeNumber"]] + distances[depotRow["NodeNumber"]][nextNodeRow["NodeNumber"]]
                next22 = depotRow["NodeNumber"]
                next32 = nextNodeRow["NodeNumber"]
            elif minPathDist is not None and minPathDist > distances[next12][depotRow["NodeNumber"]] + distances[depotRow["NodeNumber"]][nextNodeRow["NodeNumber"]]:
                minPathDist = distances[next12][depotRow["NodeNumber"]] + distances[depotRow["NodeNumber"]][nextNodeRow["NodeNumber"]]
                next22 = depotRow["NodeNumber"]
                next32 = nextNodeRow["NodeNumber"]
    path2Dist = distances[currentLoc][next31] + minPathDist




    if path2Dist >= path1Dist:
        routeList.append(next11)
        routeList.append(next21)
        routeList.append(next31)
        routeListOnlyCustomer.append(next21)
        routeListOnlyCustomer.append(next31)
        vehicleCapacityLeft = vehicleCapacity - dataframe.loc[dataframe["NodeNumber"] == next21]["Demand"] - dataframe.loc[dataframe["NodeNumber"] == next31]["Demand"]
    else:
        routeList.append(next12)
        routeList.append(next22)
        routeList.append(next32)
        routeListOnlyCustomer.append(next12)
        routeListOnlyCustomer.append(next32)
        vehicleCapacityLeft = vehicleCapacity - dataframe.loc[dataframe["NodeNumber"] == next32]["Demand"]
#Path 1:  CurrentLocation -- depot -- nextNode1 -- nextNode2. nextNode2 to be NN of nextNode1, which is NN of currentLocation
#Path 2: CurrentLocation -- nextNode1 -- depot -- nextNode2. nextNode 1 is nearest servicable node once a significant number of nodes cannot be serviced (ie capacityLeft < 8). nextNode2 is NN of depot that is not yet serviced




for index, row in depots.iterrows(): #This is a comparison-purposed iteration
    totalRouteDist = 0
    vehicleCapacityLeft = vehicleCapacity
    decidedRoute = []
    decidedRouteWithBackTrips = []
    decidedRoute.append(index)
    decidedRouteWithBackTrips.append(index)
    currentlyAt = decidedRouteWithBackTrips[-1]


    for index1, row1 in dataframe.iterrows(): #This is a repeat-purposed iteration
        minimumDist = None #Lvl 3 best
        currentlyAt = decidedRouteWithBackTrips[-1]
        if len(decidedRoute) - 1 >= nodeCount -len(depots["NodeNumber"].values):           #Stop code if all nodes are already accounted for; -1 because one depot (the starting depot) is included in decidedRoute
            break
        else:
            cannotServe = [] #Nodes in this list have demands which exceed vehicle capacity left
            minNodeCode = None
            ##Functioned-
            chooseNextNode(currentlyAt, cannotServe)
            if minNodeCode is not None: #This would be the last iteration of i
                    decidedRoute.append(minNodeCode)
                    decidedRouteWithBackTrips.append(minNodeCode)
                    vehicleCapacityLeft -= demandOfMinNode##problematic
                    try:
                        totalRouteDist += minimumDist #This try-except statement is to offset error where minimumdist is 0. Might arise in problematic execution, beware
                    except TypeError:
                        pass
                    currentlyAt = decidedRouteWithBackTrips[-1]
                    if len(decidedRoute)-1 >= nodeCount - len(depots["NodeNumber"].values):
                        if currentlyAt not in depots["NodeNumber"].to_list():
                            chooseReturnDepot(currentlyAt)
                        break
            else:
                pass








            if len(cannotServe) >= (nodeCount- len(depots) - (len(decidedRoute) -1)): #Cannot directly add trips back to decidedRoute as its len() is often used to find number of customers visited.
                calculatePath(currentlyAt, decidedRouteWithBackTrips, decidedRoute)




    if totalMinimumDist == None: ##need to edit this part to have distance be decided route with back trips
        bestRoute = decidedRouteWithBackTrips
        totalMinimumDist = totalRouteDist
    elif totalMinimumDist != None:
        if totalMinimumDist > totalRouteDist and len(bestRoute)-1 >= nodeCount-len(depots["NodeNumber"].values):
            totalMinimumDist = totalRouteDist
            bestRoute = decidedRouteWithBackTrips
        else:
            pass


depotList = [x for x in range(0, len(depots))]
def splitIntoBacktrips(depots, route,lastChopped=0, itercounter=-1): #ky is addicted to global variables. These 2 werent even used outside of this function and the defining
    backtripRoutes = [] #this one was outside? might also be used tho so JUST IN CASE i will be returning all 3 of them. 👍
    addback = route[-1]
    addback2 = route[0]
    del route[-1]
    del route[0] #Dont want to have the route be split at the first and last depot. Might as well remove for the moment.
    depotList = [x for x in range(0, len(depots))]
    for node in route:
        itercounter +=1
        if int(node) in depotList:
            backtripRoutes.append(route[lastChopped:itercounter+1])
            lastChopped = itercounter
    route.append(addback)
    backtripRoutes.append(route[lastChopped:])


    backtripRoutes2 = []
    for i in range(len(backtripRoutes)):
        if i%2 == 0:
            backtripRoutes2.append(backtripRoutes[i])
   
    backtripRoutes = backtripRoutes2
    backtripRoutes.insert(0, [addback2] + backtripRoutes[0])
    del backtripRoutes[1]
    route.append(addback)
    route.insert(0,addback2)
    return backtripRoutes


#Status: FINALLY WORKING OMGS


backtripRoutes = splitIntoBacktrips(depots, bestRoute)
print(backtripRoutes)


def intraRoute(backTripRoute, n_iterations = 1000):
    for i in range(n_iterations):
            initialDist = calculate2DDistances(backTripRoute)
            backTripRouteCopy = backTripRoute.copy()
            randomNumber = np.random.randint(1, 10*len(backTripRoute))
            randomNumber2 = np.random.randint(0, len(backTripRoute[int(randomNumber/10)]))
            randomNumber3 = np.random.randint(0, len(backTripRoute[int(randomNumber/10)]))
            while randomNumber2 == randomNumber3:
                randomNumber3 = np.random.randint(0, len(backTripRoute[int(randomNumber/10)]))
            if randomNumber2 == randomNumber3:
                randomNumber3 = np.random.randint(0, len(backTripRoute[int(randomNumber/10)]))
            swappedNode1 = backTripRoute[int(randomNumber/10)][randomNumber2]
            swappedNode2 = backTripRoute[int(randomNumber/10)][randomNumber3]
            backTripRoute[int(randomNumber/10)].insert(randomNumber2, swappedNode2)
            del backTripRoute[int(randomNumber/10)][randomNumber2+1]
            backTripRoute[int(randomNumber/10)].insert(randomNumber3, swappedNode1)
            del backTripRoute[int(randomNumber/10)][randomNumber3+1]
            finalDist = calculate2DDistances(backTripRoute)
            if initialDist > finalDist:
                pass
            elif finalDist > initialDist:
                backTripRoute = backTripRouteCopy #Revert
    return backTripRoute


def inter_route(backTripRoute):
    for i in range(len(backTripRoute)):
            for j in range(len(backTripRoute)):
                if i == j:
                    continue
                else:
                    backTripRouteCopy = backTripRoute.copy()
                    initialDist = calculate2DDistances(backTripRoute)
                    swappedRoute1 = backTripRoute[i]
                    swappedRoute2 = backTripRoute[j]
                    backTripRoute.insert(i, swappedRoute2)
                    del backTripRoute[i+1]
                    backTripRoute.insert(j, swappedRoute1)
                    del backTripRoute[j+1]
                    finalDist = calculate2DDistances(backTripRoute)
                    if initialDist < finalDist:
                        backTripRoute = backTripRouteCopy
                    else:
                        continue
    return backTripRoute


def twoOpt(backTripRoute, n_iterations = 1000, strategy = "intra_route"): #Intraroute only swaps within each backtrip. Interroute swaps entire backtrips. Rojak mode does both
    if strategy == "intra_route":
        backTripRoute = intraRoute(backTripRoute, n_iterations)
    elif strategy == "inter_route":
        backTripRoute =inter_route(backTripRoute)
    elif strategy == "rojak":
        backTripRoute =intraRoute(backTripRoute, n_iterations)
        backTripRoute =inter_route(backTripRoute)
    return backTripRoute








#Notes from lky the goat 12/6/25 ---- 3hrs ish of coding without any debugging so next person who runs it should expect a massive bug bomb lol
#13/6/25 --- debugged!


#Challenge to try creating a k-opt algorithm? ie 2-way swap, 3-way swap, 4-way swap all customizable by an input parameter.
           
       


#Plotting out route using matplotlib
routeX = []
routeY = []
nodeLabelListofRoute = []
for node in bestRoute:
    if node in depotList:
        nodeLabelListofRoute.append(0)
    else:
        nodeLabelListofRoute.append(1)


for point in bestRoute:
    routeX.append(dataframe.loc[dataframe["NodeNumber"] == point]["X"])
    routeY.append(dataframe.loc[dataframe["NodeNumber"] == point]["Y"])
plt.plot(routeX, routeY, linestyle='--', color='gray', alpha=0.5)
plt.scatter(routeX, routeY, c = nodeLabelListofRoute)
plt.show()




optimizedBacktripRoute = twoOpt(backtripRoutes, 2000, "rojak")
print(optimizedBacktripRoute)
print(totalMinimumDist)
print(nodeCount)




#Status: Working
