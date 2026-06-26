from typing import List, Optional
import collections
from collections import deque
class Solution:
    def findBottlenecks(self, n: int, edges: List[List[int]]) -> List[int]:
        graph = collections.defaultdict(list)
        needs = [0]*n
        for e in edges:
            graph[e[0]].append(e[1])
            needs[e[1]]+=1
        q = deque()
        for i in range(len(needs)):
            if needs[i] == 0:
                q.append(i)
        res = []
        process_count = 0
        while q:
            if len(q) == 1:
                res.append(q[0])
            for _ in range(len(q)):
                node = q.popleft()
                process_count+=1
                for child in graph[node]:
                    needs[child] -=1
                    if needs[child] == 0:
                        q.append(child)
                    
        return res if process_count == n else []

'''
Time Complexity: O(N + E)
Space Complexity: O(N + E)
'''


