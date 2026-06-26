intervals = [[4, 7], [10, 11], [13, 15]]
idx = 2

def remove_intervals_include(intervals, index):
    found = False
    res = []
    for interval in intervals:
        if found:
            res.append(interval)
            continue
        start, end  = interval
        length = end-start+1
        if index < length:
            found = True
            remove_point = start + index
            if remove_point > start:
                res.append([start, remove_point-1])
            if remove_point < end:
                res.append([remove_point+1, end])
        else:
            index -= length
            res.append(interval)
    return res

def remove_intervals_not_include(interval, index):
    found = False
    res = []
    for interval in intervals:
        if found:
            res.append(interval)
            continue
        start, end = interval
        length = end - start
        if index < length:
            remove_point = start+index
            if remove_point > start:
                res.append([start, remove_point])
            if remove_point+1<end:
                res.append([remove_point+1, end])
            found = True
        else:
            index -=length
            res.append(interval)
    return res


print(remove_intervals_include(intervals, idx))
print(remove_intervals_not_include(intervals, idx))