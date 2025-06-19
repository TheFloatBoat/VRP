#CSV or XLSX format: NodeNumber (MUST be unique for every single node), NodeType(e.g. 0 for depot 1 for customer), X, Y, Demand
#Use Euclidean distances for now

##VRP Solver!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print("VRP Solver!")
nodeDataframe = pd.read_csv("/content/VRPNodes115.csv")


nodeCount = len(nodeDataframe.values)
xPoints = nodeDataframe["X"].values
yPoints = nodeDataframe["Y"].values
vehicleCapacity = int(input("Input Vehicular Capacity: "))


distances = np.zeros((nodeCount, nodeCount)) #Making distance matrix
for i in range(nodeCount):
    for j in range (nodeCount):
        euclideanD = ((xPoints[j] - xPoints[i])**2 + (yPoints[j] - yPoints[i])**2)**0.5
        distances[i][j] = euclideanD
        distances[j][i] = euclideanD

startPoint = 0
depots = nodeDataframe.loc[nodeDataframe["NodeType"]==0]
totalRouteDist = 0 #Lvl 1 metric, from lvl 3 best
totalMinimumDist = None
bestRoute = None #Lvl 1 best
decidedRoute = [] #Lvl 1 competitor
decidedRouteWithBackTrips = []
minNodeCode = None


#Lvl 2 of iteration is a repeat-purposed one and does not compare anything for selection, simply has a purpose of executing lvl 3 iterations an appropriate number of times on different items


def chooseNextNodeLegacy(currentLoc, cannotServeLocal):
  #takes in the current location and the things you cant serve, edits demandOfMinNode, minNodeCode, and minimumDist

  #Nodes in cannotServe have demands which exceed vehicle capacity left
    global demandOfMinNode, minNodeCode, minimumDist    #minNodeCode is the node number of the minimum node

    for index2, row2 in nodeDataframe.iterrows():  #This is a comparison-purposed iteration
      if row2["NodeType"] != 0 and row2["NodeNumber"] != currentLoc and row2["NodeNumber"] not in decidedRoute: #Do not consider this node if it is a depot, the current node or a node that has already been visited
          #the old function has a weird inane vehicleCapacityLeft. Uncomment to see
          #print(f"before {vehicleCapacityLeft} after")
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

def chooseNextNode(currentLoc, cannotServeLocal):
  #takes in the current location and the things you cant serve, edits demandOfMinNode, minNodeCode, and minimumDist

  #Nodes in cannotServe have demands which exceed vehicle capacity left
    #variables used: demandOfMinNode, minNodeCode, minimumDist    #minNodeCode is the node number of the minimum node
    minimumDist = None
    minNodeCode = None
    for index2, row2 in nodeDataframe.iterrows():  #This is a comparison-purposed iteration
      if row2["NodeType"] != 0 and row2["NodeNumber"] != currentLoc and row2["NodeNumber"] not in decidedRoute: #Do not consider this node if it is a depot, the current node or a node that has already been visited

          #the new function has a sensible vehicleCapacityLeft. Uncomment to see
          #print(f"before {vehicleCapacityLeft} after")

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

          elif distanceToSuggested<minimumDist: #distanceToSuggested beats previous minimumDist, becomes new minimumDist.
              minimumDist = distanceToSuggested
              minNodeCode = index2
              demandOfMinNode = row2["Demand"]
          else:
              pass #aka the suggestion is longer.
    return minNodeCode, minimumDist

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


#CalculatePath functions now need to be double iterations because minimum distance is not just from currentlyAt to depot, but also then towards next node.
def calculatePath(currentLoc, routeList, routeListOnlyCustomer): #CurrentLocation -- depot -- nextNode1 -- nextNode2. nextNode2 to be NN of nextNode1, which is NN of currentLocation
    global vehicleCapacity, vehicleCapacityLeft, totalRouteDist, depots, decidedRouteWithBackTrips, distances, path1Dist, vehicleCapacity, vehicleCapacityLeft, minNodeCode
    minPathDist = None

    #finds the best depot to node combo
    for depotIndex, depotRow in depots.iterrows(): #minimum of first 3 steps
        for nextNodeIndex, nextNodeRow in nodeDataframe.iterrows():
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
    minNodeCode, minimumDist = chooseNextNode(currentLoc, cannotServe)
    path1Dist = minPathDist + minimumDist
    next31 = minNodeCode #next31 represents the third next node the vehicle would travel to if it takes path 1.
    minPathDist = None
    vehicleCapacityLeft = vehicleCapacityLeftIfNotPath1 #sets vehicleCapacityLeft to original since would not visit depot yet if takes path 2
    cannotServe = []
    #for some godsaken reason, vehicleCapacityLeft needs to be in some weird format that, for reasons beyond me(joshua), ONLY appears in the legacy function. (i did not touch the vehicle capacity at all WHY WHY WHY???)
    chooseNextNodeLegacy(currentLoc, cannotServe)
    print(type(minNodeCode))
    next12 = minNodeCode

    for depotIndex, depotRow in depots.iterrows():
        for nextNodeIndex, nextNodeRow in nodeDataframe.iterrows():
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
        vehicleCapacityLeft = vehicleCapacity - nodeDataframe.loc[nodeDataframe["NodeNumber"] == next21]["Demand"] - nodeDataframe.loc[nodeDataframe["NodeNumber"] == next31]["Demand"]
    else:
        routeList.append(next12)
        routeList.append(next22)
        routeList.append(next32)
        routeListOnlyCustomer.append(next12)
        routeListOnlyCustomer.append(next32)
        vehicleCapacityLeft = vehicleCapacity - nodeDataframe.loc[nodeDataframe["NodeNumber"] == next32]["Demand"]
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

    for index1, row1 in nodeDataframe.iterrows(): #This is a repeat-purposed iteration
        minimumDist = None #Lvl 3 best
        currentlyAt = decidedRouteWithBackTrips[-1]
        if len(decidedRoute) - 1 >= nodeCount -len(depots["NodeNumber"].values):           #Stop code if all nodes are already accounted for; -1 because one depot (the starting depot) is included in decidedRoute
            break
        else:
            cannotServe = [] #Nodes in this list have demands which exceed vehicle capacity left
            minNodeCode = None
            ##Functioned-
            chooseNextNodeLegacy(currentlyAt, cannotServe)
            if minNodeCode is not None: #This would be the last iteration of i
                    decidedRoute.append(minNodeCode)
                    decidedRouteWithBackTrips.append(minNodeCode)
                    vehicleCapacityLeft -= nodeDataframe["Demand"][minNodeCode]##problematic
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

def splitIntoBacktrips(depots, route, starterInt=0, itercounter=1):
    backtripRoutes = [] #this one was outside? might also be used tho so JUST IN CASE i will be returning all 3 of them. 👍
    addback = route[-1]
    addback2 = route[0]
    del route[-1]
    del route[0]
    depotList = depots["NodeNumber"].tolist()
    for node in route:
        itercounter +=1
        if node in depotList:
            if starterInt == 0:
                part1 = route[starterInt:itercounter]
                starterInt = itercounter
                part2 = route[itercounter: ]
                backtripRoutes.append(part1)
            else:
                part1 = part2[starterInt:itercounter]
                starterInt = itercounter
                part2 = part2[itercounter: ]
                backtripRoutes.append(part1)

    route.append(addback)
    route.append(addback2)
    print(backtripRoutes)

#Status:^^ Runs but does not work as intended. KY has given up

splitIntoBacktrips(depots, bestRoute)

#Plotting out route using matplotlib
routeX = []
routeY = []
for point in bestRoute:
    routeX.append(nodeDataframe.loc[nodeDataframe["NodeNumber"] == point]["X"])
    routeY.append(nodeDataframe.loc[nodeDataframe["NodeNumber"] == point]["Y"])
plt.plot(routeX, routeY, "--o")
plt.show()



print(bestRoute)
print(totalMinimumDist)
print(nodeCount)


#Status: Working
