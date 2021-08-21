# -*- coding: utf-8 -*-
# @Time : 2021/8/21 9:45
# @Author : cds
# @Site : https://github.com/SkyLord2?tab=repositories
# @Email: chengdongsheng@outlook.com
# @File : MonotonicQueue.py
# @Software: PyCharm
class Node:
    def __init__(self, val):
        self.val = val
        self.pre = None
        self.next = None

class LinkedList:
    def __init__(self):
        self.size = 0
        self.head = Node(None)
        self.tail = Node(None)
        self.head.next = self.tail
        self.tail.pre = self.head

    def remove(self, node: Node):
        if(self.size > 0):
            pre = node.pre
            next = node.next
            pre.next = next
            next.pre = pre
            self.size -= 1

    def append(self, n: int):
        node = Node(n)
        last = self.tail.pre
        last.next = node
        node.pre = last
        node.next = self.tail
        self.tail.pre = node
        self.size += 1
    def pop_front(self):
        first = self.head.next
        self.remove(first)
        return first

    def get_front(self):
        ret = None
        first = self.head.next
        if(first):
            ret = first.val
        return ret
    def pop_last(self):
        last = self.tail.pre
        self.remove(last)
        return last

    def get_last(self):
        last = self.tail.pre
        ret = None
        if(last):
            ret = last.val
        return ret

    def isEmpty(self):
        return self.size == 0

class MonotonicQueue:
    """
    单调队列
    """
    def __init__(self):
        self.q = LinkedList()
    def push(self, n):
        while(not self.q.isEmpty() and self.q.get_last() < n):
            self.q.pop_last()
        self.q.append(n)
    def max(self):
        return self.q.get_front()
    def pop(self, n):
        if(n == self.q.get_front()):
            self.q.pop_front()