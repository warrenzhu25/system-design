from typing import List, Optional
class Tire:
    def __init__(self):
        self.children = {}
        self.priority = float('inf')
        self.action = None

class IpFirewall:
    def __init__(self, rules: List[List[str]]):
        self.root = Tire()
        for index in range(len(rules)):
            rule = rules[index]
            action, ip= rule
            is_allow = action == 'ALLOW'
            if '/' in ip:
                part_1, part_2 = ip.split('/')
                part_2 = int(part_2)
            else:
                part_1, part_2 = ip, 32
            ip_string = self._ip_to_string(part_1)
            cur  = self.root
            for i in range(part_2):
                bit = ip_string[i]
                if bit not in cur.children:
                    cur.children[bit] = Tire()
                cur = cur.children[bit]
            if index < cur.priority:
                cur.priority = index
                cur.action = is_allow

    def _ip_to_string(self, ip):
        ip_list = ip.split('.')
        formated = (int(ip_list[0]) << 24) | (int(ip_list[1]) << 16) | (int(ip_list[2]) << 8) | (int(ip_list[3]))
        return format(formated, '032b')

    def allowAccess(self, ip: str) -> bool:
        ip_string = self._ip_to_string(ip)
        cur = self.root
        best_p = float('inf')
        best_a = False
        if  cur.priority < best_p:
            best_p = cur.priority
            best_a = cur.action

        for c in ip_string:
            if c not in cur.children:
                break
            cur = cur.children[c]
            if cur.priority < best_p:
                best_p = cur.priority
                best_a = cur.action
        
        return best_a


from typing import List, Optional
class Tire:
    def __init__(self):
        self.children = {}
        self.priority = float('inf')
        self.allow = None

        self.min_p = float('inf')
        self.min_a = None

class IpFirewall:
    def __init__(self, rules: List[List[str]]):
        self.root = Tire()
        for index in range(len(rules)):
            r = rules[index]
            allow, ip_mask = r
            is_allow = allow == 'ALLOW'
            if '/' in ip_mask:
                ip, mask = ip_mask.split('/')
            else:
                ip = ip_mask
                mask = 32
            mask = int(mask)
            ip_string  = self._ip_to_string(ip)
            node = self.root
            for i in range(mask):
                if index < node.min_p:
                    node.min_p, node.min_a = index, is_allow
                c = ip_string[i]
                if c not in node.children:
                    node.children[c] = Tire()
                node = node.children[c]
            if index < node.priority:
                node.priority = index
                node.allow = is_allow
            
            if index < node.min_p:
                node.min_p, node.min_a = index, is_allow


    def _ip_to_string(self, ip):
        ip_list = ip.split('.')
        ip_string  = (int(ip_list[0])<<24) | (int(ip_list[1])<<16) | (int(ip_list[2])<<8) | (int(ip_list[3])) 
        return format(ip_string, '032b')

    def allowAccess(self, ip_mask: str) -> bool:
        if '/' in ip_mask:
            ip, mask = ip_mask.split('/')
        else:
            ip = ip_mask
            mask = 32
        mask  = int(mask)
        node = self.root
        best_a = False
        best_p = float('inf')
        ip_string  = self._ip_to_string(ip)
        if node.priority < best_p:
            best_p = node.priority
            best_a = node.allow
        for i in range(mask):
            c = ip_string[i]
            if c in node.children:
                node = node.children[c]
                if node.priority < best_p:
                    best_p = node.priority
                    best_a = node.allow
            else:
                return best_a
        
        if node.min_p < best_p:
            best_a = node.min_a
            
        return best_a
        

