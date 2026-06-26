from typing import Dict, List, Iterator, Optional

class SnapshotSet:
    def __init__(self):
        self.version = 0
        self.look_up ={}
        self.tracker = []

    def add(self, n: int) -> bool:
        if n in self.look_up:
            return False
        self.look_up[n] = len(self.tracker)
        self.tracker.append([n,self.version, float('inf')])
        self.version+=1
        return True

    def remove(self, n: int) -> bool:
        if n not in self.look_up:
            return False
        index = self.look_up.pop(n)
        self.tracker[index][2] = self.version
        self.version+=1
        return True

    def contains(self, n: int) -> bool:
        return n in self.look_up

    def getIterator(self) -> Iterator[int]:
        return self.SnapshotIterator(self)

    class SnapshotIterator:
        def __init__(self, outer: 'SnapshotSet'):
            self.index = 0
            self.cur_version = outer.version
            self.tracker_info = outer.tracker
            self._to_valid_number()

        def __iter__(self) -> 'SnapshotSet.SnapshotIterator':
            return self

        def _to_valid_number(self):
            while self.index < len(self.tracker_info):
                number, satrt, end = self.tracker_info[self.index]
                if satrt<self.cur_version<=end:
                    break
                self.index+=1

        def __next__(self) -> int:
            if self.hasNext():
                res = self.tracker_info[self.index][0]
                self.index+=1
                self._to_valid_number()
                return res
            else:
                raise StopIteration()

        def hasNext(self) -> bool:
            self._to_valid_number()
            if self. index < len(self.tracker_info):
                return True
            else:
                return False
# Helper function to iterate all elements in the iterator for easier visualization
def iterateAllElements(it: Iterator[int]) -> List[int]:
    return list(it)

def test1():
    print("======== test 1: =========")
    s = SnapshotSet()
    print(s.add(1))  # Expected: True
    print(s.add(2))  # Expected: True
    print(s.add(3))  # Expected: True
    print(s.add(4))  # Expected: True
    print(s.add(1))  # Expected: False
    it1 = s.getIterator()
    print(s.remove(1))  # Expected: True
    print(s.remove(3))  # Expected: True
    print(s.remove(5))  # Expected: False
    it2 = s.getIterator()

    print(iterateAllElements(it1))  # Expected: [1, 2, 3, 4]
    print(iterateAllElements(it2))  # Expected: [2, 4]

def test2():
    print("======== test 2: =========")
    s = SnapshotSet()
    it1 = s.getIterator()
    print(s.add(10))  # Expected: True
    it2 = s.getIterator()
    print(s.add(20))  # Expected: True
    it3 = s.getIterator()
    print(s.add(30))  # Expected: True
    it4 = s.getIterator()
    print(s.remove(30))  # Expected: True
    it5 = s.getIterator()
    print(s.remove(20))  # Expected: True
    it6 = s.getIterator()
    print(s.remove(10))  # Expected: True
    it7 = s.getIterator()

    print(iterateAllElements(it1))  # Expected: []
    print(iterateAllElements(it2))  # Expected: [10]
    print(iterateAllElements(it3))  # Expected: [10, 20]
    print(iterateAllElements(it4))  # Expected: [10, 20, 30]
    print(iterateAllElements(it5))  # Expected: [10, 20]
    print(iterateAllElements(it6))  # Expected: [10]
    print(iterateAllElements(it7))  # Expected: []

def test3():
    print("======== test 3: =========")
    s = SnapshotSet()
    print(s.remove(5))  # Expected: False
    print(s.add(5))  # Expected: True
    print(s.remove(5))  # Expected: True
    print(s.add(5))  # Expected: True
    print(iterateAllElements(s.getIterator()))  # Expected: [5]

def test4():
    print("======== test 4: =========")
    s = SnapshotSet()
    print(s.add(1))  # Expected: True
    print(s.add(2))  # Expected: True
    print(s.add(3))  # Expected: True
    print(s.add(4))  # Expected: True
    print(s.add(5))  # Expected: True
    it1 = s.getIterator()
    print(s.remove(2))  # Expected: True
    print(s.remove(4))  # Expected: True
    print(s.add(6))  # Expected: True
    it2 = s.getIterator()
    print(iterateAllElements(it1))  # Expected: [1, 2, 3, 4, 5]
    print(iterateAllElements(it2))  # Expected: [1, 3, 5, 6]

if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()