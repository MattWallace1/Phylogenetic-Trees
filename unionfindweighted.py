# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:48:01 2022

@author: Matthew Wallace
"""



class UnionFind:
    def __init__(self,N):
        self.N = N
        self.parents = []
        self.weights = []
        for i in range(N):
            self.parents.append(i)
            self.weights.append(1)
    
    def root(self,i):
        if i != self.parents[i]:
            self.parents[i] = self.root(self.parents[i])    
        
        return self.parents[i]
    
    def find(self,i,j):
        return self.root(i) == self.root(j)
    
    
        
    
    def union(self,i,j):
        root_i = self.root(i)
        root_j = self.root(j)
        
        #print("1", self.weights[root_i])
        #print("2" , self.weights[root_j])
        if self.weights[root_i] < self.weights[root_j]:
            self.parents[root_i] = self.parents[root_j]
            self.weights[root_j] += self.weights[root_i]
        else:
            self.parents[root_j] = self.parents[root_i]
            self.weights[root_i] += self.weights[root_j]
            
        
        
if __name__ == "__main__":
    u = UnionFind(10)
    u.union(0,2)
    u.union(0,7)
    u.union(7,1)
    u.union(1,8)
    u.union(1,6)
    u.union(6,9)
    u.union(9,5)
    u.union(3,4)
    u.union(4,1)

    print(u.parents)
    print(u.weights)
    
