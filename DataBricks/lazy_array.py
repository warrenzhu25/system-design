
'''
Time Complexity
map(fn): O(1)
indexOf(target): O(N x K)

Space Complexity:
arr: O(n)
op: O(k)
memo: O(n k)
'''


# add the linked list to save the space complexity of func map from O(k) to O(1)
class OpNode:
    def __init__(self,func, pre = None):
        self.func = func
        self.pre = pre

class LazyArray:
    def __init__(self, arr, ops=None):
        self.arr = arr
        self.end_op = ops
        self.cache = {}
    
    def map(self, fn):
        cache_func = self.__cache_op__(fn)
        new_ops = OpNode(cache_func, self.end_op)
        return LazyArray(self.arr, new_ops)

    def indexOf(self, target):
        funcs = []
        ops = self.end_op
        while ops:
            funcs.append(ops.func)
            ops = ops.pre
        funcs.reverse()
        for i in range(len(self.arr)):
            if i in self.cache:
                cur = self.cache[i]
            else:
                cur = self.arr[i]
                for fn in funcs:
                    cur = fn(cur)
                
            if cur  == target:
                return i
        return -1


if __name__ == "__main__":
    # Test case 1
    arr1 = LazyArray([10, 20, 30, 40, 50])
    print(arr1.map(lambda n: n * 2).indexOf(40))  # Expected: 1

    # Test case 2
    arr2 = LazyArray([10, 20, 30, 40, 50])
    print(arr2.map(lambda n: n * 2).map(lambda n: n * 3).indexOf(240))  # Expected: 3

    # Test case 3
    arr3 = LazyArray([1, 2, 3, 4, 5])
    print(arr3.map(lambda n: n + 10).indexOf(100))  # Expected: -1

    # Test case 4
    arr4 = LazyArray([5, 10, 15, 20, 25])
    print(arr4.map(lambda n: n * 2).map(lambda n: n + 5).map(lambda n: n // 3).indexOf(11))  # Expected: 2

    # Test case 5
    arr5 = LazyArray([-5, 1, 2, -1, 10])
    print(arr5.map(lambda n: n * 3).map(lambda n: n + 4).map(lambda n: n - 2).indexOf(8))  # Expected: 2


