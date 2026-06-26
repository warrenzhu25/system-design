# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
        def dfs(node, target, path):
            if not node:
                return False
            if node.val == target:
                return True
            
            if node.left:
                path.append('L')
                if dfs(node.left, target, path):
                    return True
                path.pop()
            
            if node.right:
                path.append('R')
                if dfs(node.right, target, path):
                    return True
                path.pop()
            
            return False
        path_s = []
        path_d = []
        dfs(root, startValue, path_s)
        dfs(root, destValue, path_d)
        m = len(path_s)
        n = len(path_d)
        index = 0
        while index < min(m, n):
            if path_s[index] != path_d[index]:
                break
            index+=1
        
        return "U" * (len(path_s)-index) + "".join(path_d[index:])


# Space and Time complexity is O(n)
from typing import List, Optional
class Solution:
    def findPath(self, order: int, source: int, dest: int) -> str:
        cache= {}

        def get_cache(n):
            if n < 0:
                return 0
            if n <= 1:
                return 1
            if n in cache:
                return cache[n]
            cache[n] = 1 + get_cache(n-1) + get_cache(n-2)
            return cache[n]

        def find_path(n, root, target):
            if root == target:
                return ""
            left_root = root + 1
            left_size = get_cache(n-2)
            if left_root<= target < left_root + left_size:
                return 'L' + find_path(n-2, left_root, target)
            else:
                right_root = left_root + left_size
                return 'R' + find_path(n-1, right_root, target)
        
        source_path = find_path(order, 0, source)
        des_path = find_path(order, 0, dest)

        index = 0
        while index < min(len(source_path), len(des_path)):
            if source_path[index] != des_path[index]:
                break
            index+=1
        
        return 'U' * (len(source_path) - index) + des_path[index:]

# Pre-Order 逻辑
def find_pre(n, root_val, target):
    if root_val == target: return ""
    left_size = get_cache(n - 2)
    if target < root_val + 1 + left_size:
        return "L" + find_pre(n - 2, root_val + 1, target)
    else:
        return "R" + find_pre(n - 1, root_val + 1 + left_size, target)

# In-Order 逻辑
def find_in(n, root_val, target):
    left_size = get_cache(n - 2)
    my_root_num = root_val + left_size # 根的编号
    
    if target == my_root_num: return ""
    if target < my_root_num:
        return "L" + find_in(n - 2, root_val, target)
    else:
        return "R" + find_in(n - 1, my_root_num + 1, target)

# Post-Order 逻辑
def find_post(n, root_val, target):
    left_size = get_cache(n - 2)
    right_size = get_cache(n - 1)
    my_root_num = root_val + left_size + right_size # 根在最后
    
    if target == my_root_num: return ""
    if target < root_val + left_size:
        return "L" + find_post(n - 2, root_val, target)
    else:
        return "R" + find_post(n - 1, root_val + left_size, target)