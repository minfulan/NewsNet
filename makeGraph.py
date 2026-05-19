"""
graph function
news texts are stored in dicttionary with a ID as key.
the weights of edges are the IDs of news texts.
the numbers of vertex are IDs of news roles.


"""
import json

roleTable={"P-G-bc771-000001":"周幽王（示例）"}
eventTable={}

class Graph:
    def __init__(self,oldGraph:dict):
        self.graph = oldGraph  # 字典存储邻接关系
    
    def add_edge(self, u, v, directed=False):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        
        self.graph[u].append(v)
        if not directed:  # 无向图
            self.graph[v].append(u)
    
    def show(self):
        for node in self.graph:
            print(f"{node}: {self.graph[node]}")
