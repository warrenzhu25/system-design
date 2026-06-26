from typing import List, Optional
from collections import deque
import heapq
class Solution:
    def findOptimalCommute(self, grid: List[List[str]], modes: List[str], costs: List[int], times: List[int]) -> str:
        start = [-1, -1]
        end = [-1, -1]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 'S':
                    start = (i, j)
                elif grid[i][j] == 'D':
                    end = (i, j)
        n = len(modes)
        dirc = [(0,1), (0,-1), (-1,0), (1,0)]
        mini_cost = float('inf')
        mini_time = float('inf')
        final_modes = ''
        for index in range(n):
            target_path = str(index+1)
            q = deque()
            q.append((start, 0))
            visited = set()
            visited.add(start)
            path_cost = -1
            while q:
                node, cost = q.popleft()
                if node == end:
                    path_cost = cost
                    break
                for d in dirc:
                    x = d[0] + node[0]
                    y = d[1] + node[1]
                    if 0<=x<len(grid) and 0<=y<len(grid[0]) and (x,y) not in visited:
                        if grid[x][y] == target_path or grid[x][y] == 'D':
                            visited.add(node)
                            q.append(((x,y), cost+1))
            if path_cost != -1:
                real_cost = path_cost*costs[index]
                real_time = path_cost*times[index]
                if real_time < mini_time:
                    mini_time = real_time
                    mini_cost = real_cost
                    final_modes = modes[index]
                elif real_time  == mini_time:
                    if real_cost < mini_cost:
                        mini_cost = real_cost
                        final_modes = modes[index]
        return final_modes


class Solution:
    def findOptimalCommute(self, grid: List[List[str]], modes: List[str], costs: List[int], times: List[int]) -> List[int]:
        start = (-1, -1)
        end = (-1,-1)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 'S':
                    start = (i,j)
                if grid[i][j] == 'D':
                    end = (i,j)
        q = []
        heapq.heappush(q, (0,0,start))
        dirc = [(0,1), (0,-1), (-1, 0), (1,0)]
        best_record = {}
        best_record[start] = (0,0)
        inf_reocrd = (float('inf'), float('inf'))
        while q:
            time, cost, node = heapq.heappop(q)
            if (time,cost) > best_record.get(node, inf_reocrd):
                continue
            if node == end:
                return list(best_record[node])
            if node == start:
                cur_time = 0
                cur_cost = 0
            else:
                index = int(grid[node[0]][node[1]])-1
                cur_time = times[index]
                cur_cost = costs[index]
            for d in dirc:
                x = node[0] + d[0]
                y = node[1] + d[1]
                if 0<=x<len(grid) and 0<=y<len(grid[0]) and grid[x][y] != "X":
                    new_time = time+cur_time
                    new_cost = cost+cur_cost
                    if (new_time,new_cost) < best_record.get((x,y), inf_reocrd):
                        best_record[(x,y)] = (new_time,new_cost)
                        heapq.heappush(q,(new_time, new_cost, (x,y)))
        return [-1,-1]
                

