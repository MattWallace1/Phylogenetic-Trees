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
        
        if self.weights[root_i] < self.weights[root_j]:
            self.parents[root_i] = self.parents[root_j]
            self.weights[root_j] += self.weights[root_i]
        else:
            self.parents[root_j] = self.parents[root_i]
            self.weights[root_i] += self.weights[root_j]
