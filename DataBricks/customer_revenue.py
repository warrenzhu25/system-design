from typing import List, Optional
from sortedcontainers import SortedSet
import collections
from collections import deque

class RevenueSystem:
    def __init__(self, ):
       self.customer_id = -1
       self.id_to_revenue = {}
       self.ranking = SortedSet()
       self.children = collections.defaultdict(list)

    # O(logN)
    def add(self, revenue: int) -> int:
        self.customer_id +=1
        self.id_to_revenue[self.customer_id] = revenue
        self.ranking.add((-revenue, self.customer_id))
        return self.customer_id
    
    # O(logN)
    def addByReferral(self, revenue: int, referrerId: int) -> int:
        if referrerId not in self.id_to_revenue:
            return -1

        old_revenue = self.id_to_revenue[referrerId]
        self.ranking.remove((-old_revenue, referrerId))
        new_revenue = old_revenue + revenue
        self.ranking.add((-new_revenue, referrerId))
        self.id_to_revenue[referrerId] = new_revenue

        child = self.add(revenue)
        self.children[referrerId].append(child)
        return child
    
    # O(K)
    def getTopKCustomer(self, k: int, minRevenue: int) -> List[int]:
        res = []
        for revenue, id in self.ranking:
            revenue = -revenue
            if len(res) == k:
                break
            if revenue < minRevenue:
                break
            res.append(id)
        
        return res
    
     # O(SlogW)
    def getRelations(self, customerId: int) -> List[List[int]]:
        if customerId not in self.id_to_revenue:
            return []
        
        q = deque()
        q.append(customerId)
        res = []
        while q:
            cur = []
            for _ in range(len(q)):
                node = q.popleft()
                if node in self.children:
                    cur.extend(self.children[node])
                    for nxt in self.children[node]:
                        q.append(nxt)
            if cur: res.append(sorted(cur))
        
        return res

            
