import pandas as pd
from tqdm import tqdm

class Graph:
 
    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
 
    def DFSUtil(self, temp, v, visited):
 
        # Mark the current vertex as visited
        visited[v] = True
 
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
 
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp
 
    # method to add an undirected edge
    def addEdge(self, edge):
        v = edge[0]
        w = edge[1]
        self.adj[v].append(w)
        self.adj[w].append(v)
 
    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc

def similar_groups(similar_query_pairs: pd.DataFrame):
    V = max(similar_query_pairs.nunique().tolist())
    pair_graph = Graph(V)
    pairs_list = similar_query_pairs[[0,1]].apply(tuple, axis=1).tolist()
    print("clustering similar queries")
    for pairs in tqdm(pairs_list):
        pair_graph.addEdge(pairs)
    cc = pair_graph.connectedComponents()
    return cc