'''
https://www.1point3acres.com/interview/problems/post/7100043
1: how to optimize if the "get_load" is queried at high qps?
we can use th cache here

2: what if we want to query over variable length time?
I would use a Data Rollup (or Downsampling) strategy with Multi-Level Buckets
We maintain multiple fixed-size circular arrays with decreasing granularities. 
For recent data (last 5 minutes): We keep a 1-second granularity bucket.
For medium-term data (last 24 hours): We downsample to a 30-minute granularity bucket.
For long-term data (last 1 week): We downsample to a 12-hour granularity bucket.
'''

class HitCounter:
    def __init__(self):
        self.timestamps = []
        self.current_time = 0
        self.prefix_count = []
        self.cache = {}

    def hit(self, timestamp):
        self.current_time = max(self.current_time, timestamp)
        if len(self.timestamps) == 0:
            pre_sum = 0
        else:
            pre_sum = self.prefix_count[-1]

        if not self.timestamps or self.timestamps[-1] != timestamp:
            self.timestamps.append(timestamp)
            self.prefix_count.append(pre_sum+1)
        else:
            self.prefix_count[-1] +=1
        self.cache.clear()

    def get_load(self, seconds):
        if len(self.timestamps) == 0 or seconds <= 0:
            return 0

        if seconds in self.cache:
            return self.cache[seconds]

        target = self.current_time - seconds
        left = 0
        right = len(self.timestamps)-1
        index = -1
        while left <= right:
            mid = left + (right-left)//2
            if self.timestamps[mid] <= target:
                left = mid + 1
                index = mid
            else:
                right = mid - 1
        cut_off = self.prefix_count[index] if index != -1 else 0
        total = self.prefix_count[-1]
        self.cache[seconds] = total-cut_off
        return total-cut_off

    def get_qps(self, seconds):
        if seconds <=0:
            return 0
        return self.get_load(seconds)/seconds

class HitCounterBucket:
    def __init__(self):
        self.bucket_for_5_mins = [(0,0) for _ in range(300)]
        self.bucket_for_24_hour = [(0,0) for _ in range(24)]
        self.bucket_for_7_days = [(0,0) for _ in range(14)]

        self.current_time = 0
    
    def _update_bucket(self, buckets, size, timestamp, granularity):
        bucket_time = timestamp // granularity
        idx = bucket_time % size

        stored_time, count = buckets[idx]

        if stored_time == bucket_time:
            buckets[idx] = (stored_time, count + 1)
        else:
            buckets[idx] = (bucket_time, 1)
    
    def hit(self, timestamp):
        self.current_time = max(self.current_time, timestamp)

        self._update_bucket(self.bucket_for_5_mins, 300, timestamp, 1)
        self._update_bucket(self.bucket_for_24_hour, 24, timestamp, 3600)
        self._update_bucket(self.bucket_for_7_days, 14, timestamp, 43200)
    
    def _get_bucket(self, bucket, timestamp, granularity):
        cut_off = self.current_time - timestamp
        total = 0

        for stored_time, hits in bucket:
            actual_time = stored_time * granularity
            if actual_time > cut_off:
                total+=hits
        
        return total
        
    def get_load(self, timestamp):
        if timestamp < 0:
            return 0
        
        if timestamp <= 300:
            return self._get_bucket(self.bucket_for_5_mins, timestamp, 1)
        elif timestamp <= 86400:
            return self._get_bucket(self.bucket_for_24_hour, timestamp, 3600)
        else:
            return self._get_bucket(self.bucket_for_7_days, timestamp, 43200)    