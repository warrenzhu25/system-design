class Solution:
    def encode(self, values):
        if len(values) < 1:
            return []
        index = 0
        res = []
        n = len(values)
        while index < n:
            count = 1
            while index + count < n and values[index+count] == values[index]:
                count+=1
            
            if count >=8 or index + count == n:
                res.append(f"RLE[{values[index]}, {count}]")
                index += count
            else:
                bp_item = []
                while index < n and len(bp_item) < 8:
                    new_count = 1
                    while index + new_count < n and values[index+new_count] == values[index]:
                        new_count+=1
                    
                    if new_count>=8 or (index+new_count == n and new_count>1):
                        break
                    bp_item.append(values[index])
                    index+=1
                if bp_item:
                    res.append(f"BP{bp_item}")
        return res
                
                    
    def decode(self, runs):
        res = []
        for log in runs:
            if log.startswith("RLE"):
                info = log[4:-1]
                number, count = info.split(",")
                res += [number] * int(count)
            elif log.startswith("BP"):
                info = log[3:-1]
                numberList = info.split(",")
                for n in numberList:
                    res.append(int(n))
        return res


def test1():
    print("======== test 1: =========")
    solution = Solution()

    input = [5, 5, 5, 5, 5, 5, 5, 5, 1, 2, 3]
    encoded = solution.encode(input)
    print("encoded: " + str(encoded))
    # Expected: ["RLE[5,8]", "BP[1,2,3]"]

    decoded = solution.decode(encoded)
    print("decoded: " + str(decoded))
    # Expected: [5, 5, 5, 5, 5, 5, 5, 5, 1, 2, 3]

def test2():
    print("\n======== test 2: =========")
    solution = Solution()

    input = [1, 2, 3]
    encoded = solution.encode(input)
    print("encoded: " + str(encoded))
    # Expected: ["RLE[1,3]"]

    decoded = solution.decode(encoded)
    print("decoded: " + str(decoded))
    # Expected: [1, 1, 1]

def test3():
    print("\n======== test 3: =========")
    solution = Solution()

    input = [1, 1, 1, 1, 2, 3, 4, 5]
    encoded = solution.encode(input)
    print("encoded: " + str(encoded))
    # Expected: ["BP[1,1,1,1,2,3,4,5]"]

    decoded = solution.decode(encoded)
    print("decoded: " + str(decoded))
    # Expected: [1, 1, 1, 1, 2, 3, 4, 5]

def test4():
    print("\n======== test 4: =========")
    solution = Solution()

    input = [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    encoded = solution.encode(input)
    print("encoded: " + str(encoded))
    # Expected: ["BP[1,1,1,1,2,3,4,5]", "BP[6,7,8,9,10,11,12,13]"]

    decoded = solution.decode(encoded)
    print("decoded: " + str(decoded))
    # Expected: [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def test5():
    print("\n======== test 5: =========")
    solution = Solution()

    input = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 11]
    encoded = solution.encode(input)
    print("encoded: " + str(encoded))
    # Expected: ["RLE[0,8]", "BP[1,2,3,4,5,6,7,8]", "RLE[9,10]", "BP[10,11]"]

    decoded = solution.decode(encoded)
    print("decoded: " + str(decoded))
    # Expected: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9,
    # 9, 9, 9, 9, 10, 11]

def test6():
    print("\n======== test 6: =========")
    solution = Solution()

    input = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]
    encoded = solution.encode(input)
    print("encoded: " + str(encoded))
    # Expected: ["RLE[0,8]", "BP[1,2,3,4,5,6,7,8]", "RLE[9,3]"]

    decoded = solution.decode(encoded)
    print("decoded: " + str(decoded))
    # Expected: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]

if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()