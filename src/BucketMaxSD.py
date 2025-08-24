from sortedcontainers import SortedDict

from src.type_define import Connection, Num


class BucketMaxSD:
    def __init__(self):
        self.counts: dict[Connection, Num] = {}
        self.buckets = SortedDict() # Num->Set(Connection)

    def set(self, connection: Connection, num: Num):
        if connection not in self.counts:
            self.counts[connection] = num
            if num not in self.buckets:
                self.buckets[num] = set()
            self.buckets[num].add(connection)
            return
        # delete old one
        old_num = self.counts[connection]
        self.buckets[old_num].remove(connection)
        if len(self.buckets[old_num]) == 0:
            del self.buckets[old_num]
        # add new one
        self.counts[connection] = num
        if num not in self.buckets:
            self.buckets[num] = set()
        self.buckets[num].add(connection)


    def incr(self, conn, d=1):
        self.set(conn, self.counts.get(conn, 0) + d)

    def decr(self, conn, d=1):
        self.set(conn, self.counts.get(conn, 0) - d)


    def max_item(self) -> tuple[Connection, Num]:
        num: Num
        s: set[Connection]
        num, s = self.buckets.peekitem(-1)
        return next(iter(s)), num

