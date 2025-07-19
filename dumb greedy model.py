import pandas as pd
import math

def euclidean_distance(x1, y1, x2, y2):
    """Compute Euclidean distance between two points"""
    return math.hypot(x2 - x1, y2 - y1)

def dumb_cvrp_solver(df):
    """
    Dumb greedy solver for Capacitated VRP.
    Vehicle capacity is fixed at 200 (default for trial runs on actual algoritm as well).
    Dumb model simply services nodes in order from 0 - 114 lol
    """
    VEHICLE_CAPACITY = 200

    depot = df[df['NodeType'] == 0].iloc[0]
    depot_node = depot['NodeNumber']

    customers = df[df['NodeType'] == 1].copy()


    routes = []
    current_route = [depot_node]
    current_capacity = 0

    customers = customers.sort_values('NodeNumber')

    for _, customer in customers.iterrows():
        demand = customer['Demand']
        node = customer['NodeNumber']

        if current_capacity + demand > VEHICLE_CAPACITY:
            current_route.append(depot_node)
            routes.append(current_route)

            current_route = [depot_node, node]
            current_capacity = demand
        else:
            current_route.append(node)
            current_capacity += demand

    if len(current_route) > 1:
        current_route.append(depot_node)
        routes.append(current_route)

    return routes

def compute_route_distance(route, df):
    """Compute total distance of a single route"""
    node_pos = df.set_index('NodeNumber')[['X','Y']].to_dict('index')
    dist = 0.0
    for i in range(len(route)-1):
        a = node_pos[route[i]]
        b = node_pos[route[i+1]]
        dist += euclidean_distance(a['X'], a['Y'], b['X'], b['Y'])
    return dist

if __name__ == "__main__":
    df = pd.read_csv("C:/Users/User/VRPNodes115.csv")

    routes = dumb_cvrp_solver(df)

    total_distance = 0.0
    print("Routes and distances:")
    for i, route in enumerate(routes):
        dist = compute_route_distance(route, df)
        total_distance += dist
        print(f"Route {i+1}: {route}, Distance: {dist:.2f}")

    print(f"\nTotal distance: {total_distance:.2f}")
