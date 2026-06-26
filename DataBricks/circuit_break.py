from typing import List, Optional

# Black-box server returns deterministic boolean outcomes for each request ID.
class Server:
    def __init__(self, outcomes: List[bool]):
        self.outcomes = outcomes
        self.callCount = 0

    def handle(self, requestId: int) -> bool:
        self.callCount += 1
        if requestId < 0 or requestId >= len(self.outcomes):
            raise ValueError(f"No outcomes outcome for requestId={requestId}")
        return self.outcomes[requestId]

class CircuitBreaker:
    def __init__(self, server: Server, failureThreshold: int, resetThreshold: int):
        self.server = server
        self.failureThreshold = failureThreshold
        self.resetThreshold = resetThreshold

        self.is_open = False
        self.fail_count = 0
        self.skip_count = 0

class Gateway:
    def __init__(self, primaryBreaker: CircuitBreaker, secondaryBreaker: CircuitBreaker):
        self.primaryBreaker = primaryBreaker
        self.secondaryBreaker = secondaryBreaker

    def routeRequests(self, totalRequests: int) -> List[str]:
        res = []
        for r in range(totalRequests):
            porcess_primary = False
            process_secondary = False
            fail_primary = True
            if not self.primaryBreaker.is_open:
                porcess_primary = True
                status = self.primaryBreaker.server.handle(r)
                if status:
                    fail_primary = False
                    self.primaryBreaker.fail_count = 0
                else:
                    self.primaryBreaker.fail_count += 1
                    if self.primaryBreaker.fail_count == self.primaryBreaker.failureThreshold:
                        self.primaryBreaker.is_open = True
                        self.primaryBreaker.fail_count = 0

            else:
                self.primaryBreaker.skip_count+=1
                if self.primaryBreaker.skip_count == self.primaryBreaker.resetThreshold: 
                    self.primaryBreaker.is_open = False
                    self.primaryBreaker.skip_count = 0
            
            if (not porcess_primary) or (porcess_primary and fail_primary):
                if not self.secondaryBreaker.is_open:
                    process_secondary = True
                    status = self.secondaryBreaker.server.handle(r)
                    if status:
                        self.secondaryBreaker.fail_count = 0
                    else:
                        self.secondaryBreaker.fail_count += 1
                        if self.secondaryBreaker.fail_count == self.secondaryBreaker.failureThreshold:
                            self.secondaryBreaker.is_open = True
                            self.secondaryBreaker.fail_count = 0

                else:
                    self.secondaryBreaker.skip_count+=1
                    if self.secondaryBreaker.skip_count == self.secondaryBreaker.resetThreshold: 
                        self.secondaryBreaker.is_open = False
                        self.secondaryBreaker.skip_count = 0
            
            if porcess_primary and process_secondary:
                res.append("Primary -> Secondary")
            elif porcess_primary:
                res.append("Primary")
            elif process_secondary:
                res.append("Secondary")
            else:
                res.append("Rejected")
        return res



    @staticmethod
    def main():
        Gateway.test1()
        Gateway.test2()
        Gateway.test3()
        Gateway.test4()

    @staticmethod
    def test1():
        print("===== Test 1 =====")
        primaryOutcomes = [True, False, False, True, True, False, True]
        secondaryOutcomes = [False, True, False, False, True, True, True]

        primary = Server(primaryOutcomes)
        secondary = Server(secondaryOutcomes)

        primaryBreaker = CircuitBreaker(primary, 2, 2)
        secondaryBreaker = CircuitBreaker(secondary, 2, 2)

        gateway = Gateway(primaryBreaker, secondaryBreaker)

        print(gateway.routeRequests(7))
        # Expected: ["Primary", "Primary -> Secondary", "Primary -> Secondary",
        # "Secondary", "Rejected", "Primary", "Primary"]

    @staticmethod
    def test2():
        print("\n===== Test 2 =====")
        primaryOutcomes = [True, True, True]
        secondaryOutcomes = [False, False, False]

        primary = Server(primaryOutcomes)
        secondary = Server(secondaryOutcomes)

        primaryBreaker = CircuitBreaker(primary, 1, 2)
        secondaryBreaker = CircuitBreaker(secondary, 1, 2)

        gateway = Gateway(primaryBreaker, secondaryBreaker)

        print(gateway.routeRequests(3))
        # Expected: ["Primary", "Primary", "Primary"]

    @staticmethod
    def test3():
        print("\n===== Test 3 =====")
        primaryOutcomes = [False, True, True, False, True, False, False]
        secondaryOutcomes = [True, False, False, False, True, False, True]

        primary = Server(primaryOutcomes)
        secondary = Server(secondaryOutcomes)

        primaryBreaker = CircuitBreaker(primary, 3, 2)
        secondaryBreaker = CircuitBreaker(secondary, 1, 2)
        gateway = Gateway(primaryBreaker, secondaryBreaker)

        print(gateway.routeRequests(7))
        # Expected: ["Primary -> Secondary", "Primary", "Primary", "Primary ->
        # Secondary", "Primary", "Primary", "Primary"]

    @staticmethod
    def test4():
        print("\n===== Test 4 =====")
        primaryOutcomes = [True, False, False, True, True, False, False, True, True, True,
                          False, True]
        secondaryOutcomes = [False, True, False, False, True, True, False, False, True,
                            False, True, False]

        primary = Server(primaryOutcomes)
        secondary = Server(secondaryOutcomes)

        primaryBreaker = CircuitBreaker(primary, 2, 2)
        secondaryBreaker = CircuitBreaker(secondary, 2, 2)
        gateway = Gateway(primaryBreaker, secondaryBreaker)

        print(gateway.routeRequests(12))
        # Expected: ["Primary", "Primary -> Secondary", "Primary -> Secondary",
        # "Secondary", "Rejected", "Primary", "Primary -> Secondary", "Secondary",
        # "Rejected", "Primary", "Primary", "Primary"]

if __name__ == "__main__":
    Gateway.main()