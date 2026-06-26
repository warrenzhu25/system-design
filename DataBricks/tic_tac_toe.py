from typing import List, Optional
import collections
class TicTacToe:
    def __init__(self, n: int, m: int, k: int):
       self.log_info = collections.defaultdict(set)
       self.m = m
       self.n = n
       self.k = k
       self.winner = -1
       

    def move(self, row: int, col: int, player: int) -> int:
        if self.winner != -1 :
            return self.winner
        if row <0 or row >= self.n or col <0 or col >= self.m:
            return 0
        self.log_info[player].add((row,col))
        cur_info = self.log_info[player]
        dirc = [(0,1), (1, 0), (1, 1), (1, -1)]
        for d in dirc:
            counter = 1
            r = row + d[0]
            c = col + d[1]
            while (r, c) in cur_info:
                r += d[0]
                c += d[1]
                counter+=1
                if counter >= self.k:
                    self.winner = player
                    return player
            r = row - d[0]
            c = col - d[1]
            while (r, c) in cur_info:
                r -= d[0]
                c -= d[1]
                counter+=1
                if counter >= self.k:
                    self.winner = player
                    return player
        return 0

        
    
        
