import pytest
from ch2.BucketMaxSD import BucketMaxSD
from ch2.type_define import Connection


class TestBucketMaxSD:
    def setup_method(self):
        self.bucket = BucketMaxSD()
        
    def test_init(self):
        assert len(self.bucket.counts) == 0
        assert len(self.bucket.buckets) == 0
        
    def test_set_new_connection(self):
        conn = (b"hello", b"world")
        self.bucket.set(conn, 5)
        
        assert self.bucket.counts[conn] == 5
        assert 5 in self.bucket.buckets
        assert conn in self.bucket.buckets[5]
        
    def test_set_existing_connection(self):
        conn = (b"hello", b"world")
        
        # 首次设置
        self.bucket.set(conn, 5)
        assert self.bucket.counts[conn] == 5
        
        # 更新相同连接
        self.bucket.set(conn, 10)
        assert self.bucket.counts[conn] == 10
        assert 5 not in self.bucket.buckets  # 旧bucket应被清除
        assert conn in self.bucket.buckets[10]
        
    def test_incr_new_connection(self):
        conn = (b"a", b"b")
        self.bucket.incr(conn, 3)
        
        assert self.bucket.counts[conn] == 3
        assert conn in self.bucket.buckets[3]
        
    def test_incr_existing_connection(self):
        conn = (b"a", b"b")
        self.bucket.set(conn, 5)
        self.bucket.incr(conn, 2)
        
        assert self.bucket.counts[conn] == 7
        assert 5 not in self.bucket.buckets
        assert conn in self.bucket.buckets[7]
        
    def test_incr_default_value(self):
        conn = (b"x", b"y")
        self.bucket.incr(conn)  # 默认增加1
        
        assert self.bucket.counts[conn] == 1
        assert conn in self.bucket.buckets[1]
        
    def test_decr_existing_connection(self):
        conn = (b"test", b"data")
        self.bucket.set(conn, 10)
        self.bucket.decr(conn, 3)
        
        assert self.bucket.counts[conn] == 7
        assert 10 not in self.bucket.buckets
        assert conn in self.bucket.buckets[7]
        
    def test_decr_new_connection(self):
        conn = (b"new", b"conn")
        self.bucket.decr(conn, 2)
        
        assert self.bucket.counts[conn] == -2
        assert conn in self.bucket.buckets[-2]
        
    def test_decr_default_value(self):
        conn = (b"default", b"test")
        self.bucket.set(conn, 5)
        self.bucket.decr(conn)  # 默认减少1
        
        assert self.bucket.counts[conn] == 4
        assert conn in self.bucket.buckets[4]
        
    def test_max_item_single_connection(self):
        conn = (b"only", b"one")
        self.bucket.set(conn, 42)
        
        max_conn, max_num = self.bucket.max_item()
        assert max_conn == conn
        assert max_num == 42
        
    def test_max_item_multiple_connections(self):
        conn1 = (b"first", b"conn")
        conn2 = (b"second", b"conn")
        conn3 = (b"third", b"conn")
        
        self.bucket.set(conn1, 10)
        self.bucket.set(conn2, 25)
        self.bucket.set(conn3, 15)
        
        max_conn, max_num = self.bucket.max_item()
        assert max_conn == conn2
        assert max_num == 25
        
    def test_max_item_same_count(self):
        conn1 = (b"tie1", b"conn")
        conn2 = (b"tie2", b"conn")
        
        self.bucket.set(conn1, 20)
        self.bucket.set(conn2, 20)
        
        max_conn, max_num = self.bucket.max_item()
        assert max_conn in [conn1, conn2]  # 任一都可以
        assert max_num == 20
        
    def test_empty_bucket_max_item(self):
        with pytest.raises((KeyError, IndexError)):
            self.bucket.max_item()
            
    def test_complex_operations_sequence(self):
        conn1 = (b"alpha", b"beta")
        conn2 = (b"gamma", b"delta")
        
        # 复杂操作序列
        self.bucket.set(conn1, 10)
        self.bucket.set(conn2, 5)
        self.bucket.incr(conn2, 8)  # conn2 现在是13
        self.bucket.decr(conn1, 3)  # conn1 现在是7
        
        max_conn, max_num = self.bucket.max_item()
        assert max_conn == conn2
        assert max_num == 13
        
        # 验证状态
        assert self.bucket.counts[conn1] == 7
        assert self.bucket.counts[conn2] == 13
        assert conn1 in self.bucket.buckets[7]
        assert conn2 in self.bucket.buckets[13]
        
    def test_zero_count_bucket_cleanup(self):
        conn = (b"temp", b"conn")
        self.bucket.set(conn, 5)
        
        # 验证bucket存在
        assert 5 in self.bucket.buckets
        
        # 更新到不同值应清除旧bucket
        self.bucket.set(conn, 10)
        assert 5 not in self.bucket.buckets
        assert 10 in self.bucket.buckets