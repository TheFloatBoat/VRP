#CSV or XLSX format: NodeNumber (MUST be unique for every single node), NodeType(e.g. 0 for depot 1 for customer), X, Y, Demand
#Use Euclidean distances for now


##VRP Solver!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
print("VRP Solver!")
dataframe = pd.read_csv("C:/Users/User/VRPNodes115.csv")


#Have NN act as minimum viable product. Optimizations come separately in functions
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




def chooseNextNode(currentLoc, cannotServeLocal, vehicleCapacityLeft, data=dataframe): #Cannot serve is those which have demands that cannot fit within capacity left
  #takes in the current location and the things you cant serve, edits demandOfMinNode, minNodeCode, and minimumDist


  #Nodes in cannotServe have demands which exceed vehicle capacity left
    global demandOfMinNode, minNodeCode, minimumDist #minNodeCode is the node number of the minimum node


    for index2, row2 in data.iterrows():  #This is a comparison-purposed iteration
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


def chooseReturnDepot(currentLoc, decidedRouteWithBackTrips=decidedRouteWithBackTrips):
    global vehicleCapacity, vehicleCapacityLeft, totalRouteDist, depots, distances
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


def calculate2DDistances(backtripRoute, return1D = "False"):
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
    if return1D == "True":
        return backtripRouteDist, backtripRouteNaked
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
    chooseNextNode(currentLoc, cannotServe, vehicleCapacityLeft)
    path1Dist = minPathDist + minimumDist
    next31 = minNodeCode #next31 represents the third next node the vehicle would travel to if it takes path 1.
    minPathDist = None
    vehicleCapacityLeft = vehicleCapacityLeftIfNotPath1 #sets vehicleCapacityLeft to original since would not visit depot yet if takes path 2
    cannotServe = []
    chooseNextNode(currentLoc, cannotServe,vehicleCapacityLeft)
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
            chooseNextNode(currentlyAt, cannotServe, vehicleCapacityLeft)
            if minNodeCode is not None: #This would be the last iteration of i
                    decidedRoute.append(minNodeCode)
                    decidedRouteWithBackTrips.append(minNodeCode)
                    vehicleCapacityLeft -= demandOfMinNode
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




    if totalMinimumDist == None: 
        bestRouteNN = decidedRouteWithBackTrips
        totalMinimumDist = totalRouteDist
    elif totalMinimumDist != None:
        if totalMinimumDist > totalRouteDist and len(bestRouteNN)-1 >= nodeCount-len(depots["NodeNumber"].values):
            totalMinimumDist = totalRouteDist
            bestRouteNN = decidedRouteWithBackTrips
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


backtripRoutes = splitIntoBacktrips(depots, bestRouteNN)
print(backtripRoutes)




#Optimization
def intraRoute(backTripRoute, n_iterations = 1000):
    global bestRoute
    for i in range(n_iterations):
            initialDist, initial1D = calculate2DDistances(backTripRoute, return1D = "True")
            backTripRouteCopy = backTripRoute.copy()
            randomNumber = np.random.randint(1, 10*len(backTripRoute))
            randomNumber2 = np.random.randint(1, len(backTripRoute[int(randomNumber/10)])-1)
            randomNumber3 = np.random.randint(1, len(backTripRoute[int(randomNumber/10)])-1)
            while randomNumber2 == randomNumber3:
                randomNumber3 = np.random.randint(1, len(backTripRoute[int(randomNumber/10)])-1)
            if randomNumber2 == randomNumber3:
                randomNumber3 = np.random.randint(1, len(backTripRoute[int(randomNumber/10)])-1)
            swappedNode1 = backTripRoute[int(randomNumber/10)][randomNumber2]
            swappedNode2 = backTripRoute[int(randomNumber/10)][randomNumber3]
            backTripRoute[int(randomNumber/10)].insert(randomNumber2, swappedNode2)
            del backTripRoute[int(randomNumber/10)][randomNumber2+1]
            backTripRoute[int(randomNumber/10)].insert(randomNumber3, swappedNode1)
            del backTripRoute[int(randomNumber/10)][randomNumber3+1]
            finalDist, final1D = calculate2DDistances(backTripRoute, return1D = "True")
            if initialDist > finalDist:
                bestRoute = initial1D
            elif finalDist > initialDist:
                backTripRoute = backTripRouteCopy #Revert
                bestRoute = final1D
    return backTripRoute


def inter_route(backTripRoute):
    global bestRoute
    for i in range(len(backTripRoute)):
            for j in range(len(backTripRoute)):
                if i == j:
                    continue
                else:
                    backTripRouteCopy = backTripRoute.copy()
                    initialDist, initial1D = calculate2DDistances(backTripRoute, return1D = "True")
                    swappedRoute1 = backTripRoute[i]
                    swappedRoute2 = backTripRoute[j]
                    backTripRoute.insert(i, swappedRoute2)
                    del backTripRoute[i+1]
                    backTripRoute.insert(j, swappedRoute1)
                    del backTripRoute[j+1]
                    finalDist, final1D = calculate2DDistances(backTripRoute, return1D = "True")
                    if initialDist < finalDist:
                        backTripRoute = backTripRouteCopy
                        bestRoute = final1D
                    else:
                        bestRoute = initial1D
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



#Useful tip: set(list) returns all unique values in list. didnt know lol


def clusterMidpoint(clusterXYDF):
    totalX = 0
    totalY = 0
    for i in range(len(clusterXYDF)):
        totalX += clusterXYDF.iloc[i]["X"]
        totalY += clusterXYDF.iloc[i]["Y"]
   
    return [totalX/len(clusterXYDF), totalY/len(clusterXYDF)] #[X, Y]


def clusterOrders(clusteredList):
    clusterOrderList = []
    clusterMidpointList = []
    for i in range(len(clusteredList)):
        clusterMidpointList.append(clusterMidpoint(dataframe.loc[dataframe["NodeNumber"].isin(clusteredList[i])][["X","Y"]]))
    clusterMidpointDict = dict(zip([x for x in range(len(clusteredList))], clusterMidpointList))
    clusterOrderList.append(0) #This would be the cluster of index 0 in clusteredList
    for i in range(len(clusteredList)-1):
        minClustDist = None
        nearestCluster = None
        currentCluster = clusterOrderList[-1]
        currentClusterCoors = clusterMidpointDict.get(currentCluster)
        for key, value in clusterMidpointDict.items():
            potentialNextCoors = value
            if key in clusterOrderList:
                continue
            if minClustDist == None:
                minClustDist = ((currentClusterCoors[0] - potentialNextCoors[0])**2 + (currentClusterCoors[1] - potentialNextCoors[1])**2)**0.5
                nearestCluster = key
            elif minClustDist > ((currentClusterCoors[0] - potentialNextCoors[0])**2 + (currentClusterCoors[1] - potentialNextCoors[1])**2)**0.5:
                minClustDist = ((currentClusterCoors[0] - potentialNextCoors[0])**2 + (currentClusterCoors[1] - potentialNextCoors[1])**2)**0.5
                nearestCluster = key
       
        clusterOrderList.append(nearestCluster)
   
    return clusterOrderList #This list would always start with the first cluster (cluster 0)

def anyDistance(node1, node2):
    x1 = node1[0]
    y1 = node1[1]
    x2 = node2[0]
    y2 = node2[1]
    return ((x1 - x2)**2 + (y1-y2)**2)**0.5

#Technically this dbscan optimization can also be used as a minimum viable product since it generates a whole route to be  compared with that of NN
def clusterRouted(epsilon, min_samples, noisePoints = "Address", coordinates = dataframe[["X","Y"]], vehicleCapacity = vehicleCapacity, depotList = depotList, distances = distances):
    global totalRouteDist #Guys pls dont change this i rlly need this so that an above function can modify an internal variable here
    dbscanModel = DBSCAN(eps = epsilon, min_samples= min_samples)
    dbscanModel.fit(coordinates)
    catList = dbscanModel.labels_
    catsDict = dict(zip(dataframe["NodeNumber"], catList))
    collectiveClusterRoute = []
    if -1 in catList:
        nunique = len(set(catList)) - 1
    else:
        nunique = len(set(catList))
    clusteredList = []
    for i in range(nunique):
        list2 = []
        for key, value in catsDict.items(): #Oof apparently you cant just do key, value in catsDict, need to .items()
            if value == i:
                if key in depotList:
                    pass
                else:
                    list2.append(key)
        clusteredList.append(list2) #2D List of all customer nodes, dimensioned by cluster of dbscan
    #Starting depot will be depot closest to a core node
    coreNodesList = dbscanModel.core_sample_indices_
    minNodeToDepot = None
    for i in range(len(depots)):
        for j in coreNodesList:
            if j in depotList:
                continue
            if minNodeToDepot == None:
                startDepot = i
                startNode = j
                minNodeToDepot = distances[i][j]
            else:
                if minNodeToDepot > distances[i][j]:
                    startDepot = i
                    startNode = j
                    minNodeToDepot = distances[i][j]
                else:
                    continue
    totalRouteDist = 0
    vehicleCapacityLeft = vehicleCapacity
    iterno = -1
    for i in clusterOrders(clusteredList): 
        iterno+=1
        if i == 0:
            collectiveClusterRoute.append(startDepot)
            collectiveClusterRoute.append(startNode)
            totalRouteDist += distances[startDepot][startNode]
            x = len(coordinates.iloc[[x for x in clusteredList[i]]]) -1
        coordinateSet = dataframe.iloc[[x for x in clusteredList[i]]]
        x = len(coordinateSet)
        clusterVisitLiao = []
        for j in range(x):
            currentlyAt = collectiveClusterRoute[-1]
            minNodeDist = None
            for node in coordinateSet["NodeNumber"].to_list():
                if node in collectiveClusterRoute:
                    continue
                if node in depotList:
                    continue
                if node in clusterVisitLiao:
                    continue
                if (coordinateSet.loc[coordinateSet["NodeNumber"] == node]["Demand"].values > vehicleCapacityLeft).any():
                    chooseReturnDepot(currentlyAt, decidedRouteWithBackTrips=collectiveClusterRoute)
                    currentlyAt = collectiveClusterRoute[-1]
                    minNodeDist = None
                if minNodeDist == None:
                    minNodeDist = distances[node][currentlyAt]
                    closestNode = node
                elif minNodeDist > distances[node][currentlyAt]:
                    minNodeDist = distances[node][currentlyAt]
                    closestNode = node

            try:
                totalRouteDist += minNodeDist
                vehicleCapacityLeft -= coordinateSet.loc[coordinateSet["NodeNumber"] == closestNode]["Demand"]
                collectiveClusterRoute.append(closestNode)
                clusterVisitLiao.append(closestNode)
            except:
                pass


        if (vehicleCapacityLeft > 0.5*vehicleCapacity).any():
            pass
        else:
            try: #There will be no next cluster if it is the last cluster being serviced
                nextClusterMP = clusterMidpoint(dataframe.loc[dataframe["NodeNumber"].isin(clusteredList[clusterOrders(clusteredList)[iterno+1]])][["X","Y"]]) #Takes time to understand yeah. Basically applies clusterMidpoint() and passes in the XY coordinate dataframe of all nodes in the NEXT cluster
                minVar = None
                for depot in depotList:
                    if minVar == None or minVar > (anyDistance(dataframe.loc[dataframe["NodeNumber"] == currentlyAt][["X","Y"]].to_list(), dataframe.loc[dataframe["NodeNumber"] == depot][["X","Y"]].to_list()) + anyDistance(dataframe.loc[dataframe["NodeNumber"] == depot][["X","Y"]].to_list(), nextClusterMP)):
                        minVar = anyDistance(dataframe.loc[dataframe["NodeNumber"] == currentlyAt][["X","Y"]].values.tolist(), dataframe.loc[dataframe["NodeNumber"] == depot][["X","Y"]].values.tolist()) + anyDistance(dataframe.loc[dataframe["NodeNumber"] == depot][["X","Y"]].to_list(), nextClusterMP)
                        bestDepot = depot
                    else:
                        continue
            except IndexError:
                continue

            vehicleCapacityLeft = vehicleCapacity
            totalRouteDist += minVar
            collectiveClusterRoute.append(bestDepot)

    noiseNodes = [dataframe.iloc[index]["NodeNumber"] for index, category in enumerate(catList) if category == -1] #Wow i just realised enumerate(list) returns [(index, element)]
    if noisePoints == "Ignore":
        chooseReturnDepot(collectiveClusterRoute[-1], decidedRouteWithBackTrips=collectiveClusterRoute)
        nodesDitched = len(noiseNodes)
    elif noisePoints == "Address":
        nodesDitched = 0
        currentlyAt = collectiveClusterRoute[-1]
        for i in range(len(noiseNodes)):
            minVar = None
            best = None
            for j in noiseNodes:
                if j in collectiveClusterRoute:
                    continue
                if (dataframe.loc[dataframe["NodeNumber"] == j]["Demand"].values > vehicleCapacityLeft).any():
                    chooseReturnDepot(currentlyAt, decidedRouteWithBackTrips=collectiveClusterRoute)
                if minVar == None or minVar > distances[currentlyAt][j]:
                    minVar = distances[currentlyAt][j]
                    best = j
                else:
                    continue
            try:
                totalRouteDist += minVar
                vehicleCapacityLeft -= dataframe.loc[dataframe["NodeNumber"] == best]["Demand"]
                collectiveClusterRoute.append(best)
            except:
                continue
    print("Nodes Unserviced:", nodesDitched)
    return {"route": collectiveClusterRoute, "distance": totalRouteDist}
         
            #Right now (13/6/25 update) nothing is stopping the algorithm from carrying out nearest neighbour after the first iteration.
            #Things to add/improve:
            #1) After one cluster is complete, go back to strategic depot such that currentlyAt --- depot --- next cluster is minimized. Remember to refill capacity
            #2) Choose order of clusters by nearest neighbour, but for entire cluser. ie take midpoint of all points in each cluster, then nearest neighbour the midpoints to find out cluster order
            #3) Address points that are OUTSIDE of the cluster (noise points) (the category of these points are labelled as -1) (they do not count as a category in nunique)
            #14/6/25 --- DONE!
             
dict1 = clusterRouted(10, 4, noisePoints="Address", vehicleCapacity=vehicleCapacity)

#Plotting out route using matplotlib
routeX = []
routeY = []
nodeLabelListofRoute = []


for node in bestRouteNN:
    if node in depotList:
        nodeLabelListofRoute.append(0)
    else:
        nodeLabelListofRoute.append(1)

for point in bestRouteNN:
    routeX.append(dataframe.loc[dataframe["NodeNumber"] == point]["X"])
    routeY.append(dataframe.loc[dataframe["NodeNumber"] == point]["Y"])
plt.plot(routeX, routeY, linestyle='--', color='gray', alpha=0.5)
plt.scatter(routeX, routeY, c = nodeLabelListofRoute) #This does not plot the ditched points for dbscan
plt.show()


optimizedBacktripRoute = twoOpt(backtripRoutes, 2000, "rojak")
print(optimizedBacktripRoute)
print(calculate2DDistances(optimizedBacktripRoute))
print(nodeCount)
print(dict1)

#Challenge to have vehicles serve noise points while transiting from node to depot if on the way.
#Challenge to try creating a k-opt algorithm? ie 2-way swap, 3-way swap, 4-way swap all customizable by an input parameter.

#Color the nodes by dbscan cluster on a separate scatter plot

#Status: Working!
