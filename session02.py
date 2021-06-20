# -*- coding: utf-8 -*-
# @Time : 2021/1/30 16:06
# @Author : cds
# @Site : https://github.com/SkyLord2?tab=repositories
# @Email: chengdongsheng@outlook.com
# @File : session02.py
# @Software: PyCharm

import sys
import traceback
from typing import List
from queue import Queue
from queue import PriorityQueue
from collections import defaultdict
import heapq
import cmath
import random
import functools
import bisect

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

"""
297. 二叉树的序列化与反序列化
序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，
同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，
你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。
提示: 输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅 LeetCode 序列化二叉树的格式。
你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。s
"""
class Codec:
    def __init__(self):
        self.res = ''

    def serialize(self, root):
        """Encodes a tree to a single string.
        前序遍历
        :type root: TreeNode
        :rtype: str
        """
        def serial(root):
            if(not root):
                self.res = self.res + '#' + ','
                return
            self.res = self.res + str(root.val) + ','
            self.serialize(root.left)
            self.serialize(root.right)
        serial(root)
        return self.res


    def deserialize(self, data):
        """Decodes your encoded data to tree.
        :type data: str
        :rtype: TreeNode
        """
        sdata = data.split(',')
        def deserial(sdata):
            if(not sdata):
                return None
            s = sdata[0]
            del(sdata[0])
            if(s == '#'):
                return None
            root = TreeNode(int(s))
            root.left = deserial(sdata)
            root.right = deserial(sdata)
            return root
        return deserial(sdata)

"""
341. 扁平化嵌套列表迭代器
给你一个嵌套的整型列表。请你设计一个迭代器，使其能够遍历这个整型列表中的所有整数。

列表中的每一项或者为一个整数，或者是另一个列表。其中列表的元素也可能是整数或是其他列表。
"""
class NestedInteger:
   def isInteger(self) -> bool:
       """
       @return True if this NestedInteger holds a single integer, rather than a nested list.
       """

   def getInteger(self) -> int:
       """
       @return the single integer that this NestedInteger holds, if it holds a single integer
       Return None if this NestedInteger holds a nested list
       """

   def getList(self): # -> [NestedInteger]:
       """
       @return the nested list that this NestedInteger holds, if it holds a nested list
       Return None if this NestedInteger holds a single integer
       """
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.res = []
        for ele in nestedList:
            self.traverse(ele, self.res)

    def next(self) -> int:
        first = self.res[0]
        del(self.res[0])
        return first
    def hasNext(self) -> bool:
        return len(self.res) != 0
    def traverse(self, nestedList, res):
        if(nestedList.isInteger()):
            res.append(nestedList.getInteger())
            return
        for ele in nestedList:
            self.traverse(ele, res)

class NestedIteratorEx:
    def __init__(self, nestedList: [NestedInteger]):
        self.retlist = nestedList

    def next(self) -> int:
        first = self.retlist[0].getInteger()
        del (self.retlist[0])
        return first

    def hasNext(self) -> bool:
        while (self.retlist and not self.retlist[0].isInteger()):
            first = self.retlist[0].getList();
            del (self.retlist[0])
            for i in range(len(first) - 1, -1, -1):
                item = first[i]
                self.retlist.insert(0, item)
        return len(self.retlist) != 0
"""
查并集
主要解决图论中的动态连通性问题，
满足：
1. 自反性
2. 对称性
3. 传递性
"""
class UnionFind:
    def __init__(self, n):
        self._count = n
        # self.parent[i] 表示节点 i 的父节点
        self.parent = [0 for i in range(n)]
        # self.size[i] 表示节点 i 的子节点的个数
        self.size = [0 for i in range(n)]
        # 初始，节点只和自己连通
        for i in range(n):
            self.parent[i] = i
            self.size[i] = 1
    def union(self, p, q):
        """
        将点p 与 点q 联通在一起
        :param p:
        :param q:
        :return:
        """
        rootP = self.find(p)
        rootQ = self.find(q)
        if(rootP == rootQ):
            return
        # 保持树的平衡性
        if(self.size[rootP] > self.size[rootQ]):
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        self._count -= 1
    def find(self, p):
        """
        得到节点p的根节点
        :param p:
        :return:
        """
        while(self.parent[p] != p):
            # 压缩树的高度
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p
    def connected(self, p, q):
        """
        点p 与 点q是否连通
        :param p:
        :param q:
        :return:
        """
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ
    def count(self):
        """
        返回当前节点集中的可联通分量
        :return:
        """
        return self._count

"""
146. LRU 缓存机制
运用你所掌握的数据结构，设计和实现一个   LRU (最近最少使用) 缓存机制 。
实现 LRUCache 类：

LRUCache(int capacity) 以正整数作为容量  capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value)  如果关键字已经存在，则变更其数据值；
如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
进阶：你是否可以在  O(1) 时间复杂度内完成这两种操作？
"""
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.pre = None
        self.next = None

class LinkedList:
    def __init__(self):
        self.size = 0
        self.head = Node(None, None)
        self.tail = Node(None, None)
        self.head.next = self.tail
        self.tail.pre = self.head

    def remove(self, node: Node):
        if(self.size > 0):
            pre = node.pre
            next = node.next
            pre.next = next
            next.pre = pre
            self.size -= 1

    def append(self, node: Node):
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

class LRUCache:
    """
    1. 要求删除最久未使用，需要保持 插入 顺序， 使用链表
    2. 要求 O(1) 时间复杂度，使用 Hash
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.k = LinkedList()
        self.kv = {}

    def get(self, key: int) -> int:
        if(key in self.kv):
            node = self.kv[key]
            self.k.remove(node)
            self.k.append(node)
            return node.val
        else:
            return -1
    def put(self, key: int, value: int) -> None:
        if(key in self.kv):
            node = self.kv[key]
            node.val = value
            self.k.remove(node)
            self.k.append(node)
        else:
            if(len(self.kv) >= self.capacity):
                first = self.k.pop_front()
                del self.kv[first.key]
            new_node = Node(key, value)
            self.k.append(new_node)
            self.kv[key] = new_node

"""
460. LFU 缓存
请你为 最不经常使用（LFU）缓存算法设计并实现数据结构。

实现 LFUCache 类：
    1.LFUCache(int capacity) - 用数据结构的容量 capacity 初始化对象
    2.int get(int key) - 如果键存在于缓存中，则获取键的值，否则返回 -1。
    3.void put(int key, int value) - 如果键已存在，则变更其值；
        如果键不存在，请插入键值对。当缓存达到其容量时，则应该在插入新项之前，使最不经常使用的项无效。
        在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，应该去除 最久未使用 的键。
    4.注意「项的使用次数」就是自插入该项以来对其调用 get 和 put 函数的次数之和。使用次数会在对应项被移除后置为 0 。

为了确定最不常使用的键，可以为缓存中的每个键维护一个 使用计数器 。使用计数最小的键是最久未使用的键。
当一个键首次插入到缓存中时，它的使用计数器被设置为 1 (由于 put 操作)。对缓存中的键执行 get 或 put 操作，使用计数器的值将会递增。
"""


class LinkedHashSet:
    def __init__(self):
        self.l = []
        self.s = set()

    def add(self, val):
        if (val not in self.s):
            self.l.append(val)
            self.s.add(val)
        return self

    def get(self, idx):
        if (idx >= len(self.l)):
            return None
        else:
            return self.l[idx]

    def pop(self):
        last = self.get(-1)
        self.remove(last)
        return last

    def remove(self, val):
        if (val in self.s):
            self.s.remove(val)
            self.l.remove(val)

    def size(self):
        return len(self.l)

    def existed(self, val):
        return (val in self.s)


class LFUCache:

    def __init__(self, capacity: int):
        self.keyToVal = {}
        self.keyToFreq = {}
        self.freqToKeys = {}
        self.minFreq = 0
        self.capacity = capacity

    def get(self, key: int) -> int:
        if (key in self.keyToVal):
            self.increaseFreq(key)
            return self.keyToVal[key]
        else:
            return -1

    def increaseFreq(self, key):
        if (key in self.keyToFreq):
            freq = self.keyToFreq[key]
            oringinKeys = self.freqToKeys[freq]
            oringinKeys.remove(key)

            if (oringinKeys.size() == 0):
                del self.freqToKeys[freq]
                if (freq == self.minFreq):
                    self.minFreq = freq + 1

            freq += 1
            if (freq in self.freqToKeys):
                self.freqToKeys[freq].add(key)
            else:
                self.freqToKeys[freq] = LinkedHashSet().add(key)

            self.keyToFreq[key] = freq
        else:
            self.keyToFreq[key] = 1
            if (1 in self.freqToKeys):
                self.freqToKeys[1].add(key)
            else:
                self.freqToKeys[1] = LinkedHashSet().add(key)
            self.minFreq = 1

    def removeMinFreq(self):
        keys = self.freqToKeys[self.minFreq]
        oldKey = keys.get(0)
        keys.remove(oldKey)
        if (keys.size() == 0):
            del self.freqToKeys[self.minFreq]

        del self.keyToFreq[oldKey]
        del self.keyToVal[oldKey]

    def put(self, key: int, value: int) -> None:
        if (self.capacity <= 0):
            return
        if (key in self.keyToVal):
            self.keyToVal[key] = value
            self.increaseFreq(key)
        else:
            if (len(self.keyToVal) < self.capacity):
                pass
            else:
                # 清除使用频率最小的键值
                self.removeMinFreq()
            self.keyToVal[key] = value
            self.increaseFreq(key)

"""
895. 最大频率栈
实现 FreqStack，模拟类似栈的数据结构的操作的一个类。

FreqStack 有两个函数：

push(int x)，将整数 x 推入栈中。
pop()，它移除并返回栈中出现最频繁的元素。
如果最频繁的元素不只一个，则移除并返回最接近栈顶的元素。
"""
class FreqStack:

    def __init__(self):
        self.max_freq = 0   # 出现频率最大的元素
        self.vf = {}        # 各个元素对应的出现频率
        self.fvs = {}       # 不同的出现频率对应的元素

    def push(self, val: int) -> None:
        if(val in self.vf):
            old_freq = self.vf[val]
            self.vf[val] = old_freq + 1
        else:
            self.vf[val] = 1

        freq = self.vf[val]

        if(freq > self.max_freq):
            self.max_freq = freq

        if(freq in self.fvs):
            self.fvs[freq].append(val)
        else:
            li = []
            li.append(val)
            self.fvs[freq] = li


    def pop(self) -> int:
        vals = self.fvs[self.max_freq]
        last = vals.pop()
        freq = self.vf[last] - 1
        self.vf[last] = freq
        if(len(vals) == 0):
            self.max_freq -= 1
        return last

"""
295. 数据流的中位数
中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。

例如，
[2,3,4]的中位数是 3
[2,3] 的中位数是 (2 + 3) / 2 = 2.5
设计一个支持以下两种操作的数据结构：
void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
"""
class MaxHeap(object):
    """
    大顶堆
    """
    def __init__(self):
        self.heap = []
        heapq.heapify(self.heap)
    def push(self, value):
        heapq.heappush(self.heap, -value)
    def pop(self):
        return -heapq.heappop(self.heap)
    def getMax(self):
        return - self.heap[0]
    def size(self):
        return len(self.heap)

class MinHeap(object):
    """
    小顶堆
    """
    def __init__(self):
        self.heap = []
        heapq.heapify(self.heap)
    def push(self, value):
        heapq.heappush(self.heap, value)
    def pop(self):
        return heapq.heappop(self.heap)
    def getMin(self):
        return self.heap[0]
    def size(self):
        return len(self.heap)

class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.minHeap = MinHeap()
        self.maxHeap = MaxHeap()

    def addNum(self, num: int) -> None:
        minSize = self.minHeap.size()
        maxSize = self.maxHeap.size()
        if(minSize >= maxSize):
            self.minHeap.push(num)
            self.maxHeap.push(self.minHeap.pop())
        else:
            self.maxHeap.push(num)
            self.minHeap.push(self.maxHeap.pop())

    def findMedian(self) -> float:
        minSize = self.minHeap.size()
        maxSize = self.maxHeap.size()
        if(minSize > maxSize):
            return self.minHeap.getMin()
        elif(maxSize > minSize):
            return self.maxHeap.getMax()
        else:
            return (self.minHeap.getMin() + self.maxHeap.getMax())/2

"""
355. 设计推特
设计一个简化版的推特(Twitter)，可以让用户实现发送推文，关注/取消关注其他用户，能够看见关注人（包括自己）的最近十条推文。你的设计需要支持以下的几个功能：

postTweet(userId, tweetId): 创建一条新的推文
getNewsFeed(userId): 检索最近的十条推文。每个推文都必须是由此用户关注的人或者是用户自己发出的。推文必须按照时间顺序由最近的开始排序。
follow(followerId, followeeId): 关注一个用户
unfollow(followerId, followeeId): 取消关注一个用户
"""
class Twitter:
    timestamp = 0
    class Tweet:
        def __init__(self, id, time):
            # tweet的id
            self.id = id
            # tweet的发布时间
            self.time = time
            # 下一条推特
            self.next = None
        def __lt__(self, other):
            """
            降序排列
            :param other:
            :return:
            """
            return self.time > other.time

    class User:
        def __init__(self, userId):
            # 用户的Id
            self.userId = userId
            # 用户的关注列表
            self.followed = set()
            # 用户发布的tweet
            self.head = None
            # 用户关注自己
            self.follow(userId)

        def follow(self, userId):
            self.followed.add(userId)

        def unfollow(self, userId):
            # 不能取关自己
            if(self.userId != userId):
                if(userId in self.followed):
                    self.followed.remove(userId)
        def post(self, id):
            tweet = Twitter.Tweet(id, Twitter.timestamp)
            Twitter.timestamp += 1
            # 越新的tweet越靠前
            tweet.next = self.head
            # 最新的tweet作为头节点
            self.head = tweet

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.userMap = {}

    def postTweet(self, userId: int, tweetId: int) -> None:
        """
        Compose a new tweet.
        """
        if(userId not in self.userMap):
            user = Twitter.User(userId)
            self.userMap[userId] = user
        user = self.userMap[userId]
        user.post(tweetId)

    def getNewsFeed(self, userId: int) -> List[int]:
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        """
        if(userId not in self.userMap):
            return []
        pq = PriorityQueue()
        ret = []
        user = self.userMap[userId]
        for followedId in user.followed:
            followee = self.userMap[followedId]
            if(followee.head):
                pq.put(followee.head)

        while(not pq.empty()):
            if(len(ret) == 10):
                break
            tweet = pq.get()
            ret.append(tweet.id)
            if(tweet.next):
                pq.put(tweet.next)

        return ret



    def follow(self, followerId: int, followeeId: int) -> None:
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        """
        if (followerId not in self.userMap):
            follower = Twitter.User(followerId)
            self.userMap[followerId] = follower
        if (followeeId not in self.userMap):
            followee = Twitter.User(followeeId)
            self.userMap[followeeId] = followee
        follower = self.userMap[followerId]
        follower.follow(followeeId)


    def unfollow(self, followerId: int, followeeId: int) -> None:
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        """
        if (followerId in self.userMap):
            follower = self.userMap[followerId]
            follower.unfollow(followeeId)

class Solution:
    def __init__(self):
        self.memo = {}

    """
    509. 斐波那契数
    斐波那契数，通常用  F(n) 表示，形成的序列称为 斐波那契数列 。该数列由  0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：

    F(0) = 0，F(1)  = 1
    F(n) = F(n - 1) + F(n - 2)，其中 n > 1
    给你 n ，请计算 F(n) 。
    """
    def fib(self, n: int) -> int:
        dp = [-1 for _ in range(n + 1)]
        # 边界条件
        dp[0] = 0
        if (n < 1):
            return dp[0]
        dp[1] = 1

        for i in range(2, n + 1):
            # 状态转移
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n]
    """
    322. 零钱兑换
    给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。
    如果没有任何一种硬币组合能组成总金额，返回 -1。
    你可以认为每种硬币的数量是无限的。
    """
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp[i]表示凑成金额 i 所需要的最少的硬币个数
        dp = [float('inf') for _ in range(amount + 1)]
        # 边界条件
        dp[0] = 0

        for i in range(1, amount + 1):
            # 状态转移
            if(i in coins):
                dp[i] = 1
            else:
                for c in coins:
                    if((i-c) >= 0):
                        dp[i] = min(dp[i-c] + 1, dp[i])
        return dp[amount] if(dp[amount] != float('inf')) else -1
    """
    46. 全排列
    给定一个 没有重复 数字的序列，返回其所有可能的全排列。
    
    示例：
    输入: [1,2,3]
    输出:
    [
      [1,2,3],
      [1,3,2],
      [2,1,3],
      [2,3,1],
      [3,1,2],
      [3,2,1]
    ]
    """
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        track = []
        def recall(nums: List[int], track: List[int]):
            # 遍历到底了，添加结果
            if (len(track) == len(nums)):
                res.append(track.copy())
                return
            for i in range(0, len(nums)):
                if (nums[i] in track):
                    continue
                # 选择当前的数字
                track.append(nums[i])
                # 继续向后选择
                recall(nums, track)
                # 取消当前的选择，做另外的选择
                track.remove(nums[i])
        recall(nums, track)
        return res
    """
    51. N 皇后
    n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
    给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。
    每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
    皇后彼此不能相互攻击，也就是说：任何两个皇后都不能处于同一条横行、纵行或斜线上。
    1 <= n <= 9
    """
    def solveNQueens(self, n: int) -> List[List[str]]:
        board = [['.' for i in range(n)] for j in range(n)]
        res = []
        def isValid(board: List[List[str]], row: int, col: int):
            nRow = len(board)
            nCol = len(board)
            # 检查列
            for r in range(nRow):
                if(board[r][col] == 'Q'):
                    return False
            # 检查右上
            for r, c in zip(range(row-1, -1, -1), range(col+1, nCol)):
                if(board[r][c] == 'Q'):
                    return False
            # 检查左上
            for r, c in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
                if(board[r][c] == 'Q'):
                    return False
            return True
        def turnToStr(board: List[List[str]]) -> List[str]:
            sList = []
            for idx, l in enumerate(board):
                sList.append(''.join(l))
            return sList

        def recall(board: List[List[str]], row: int):
            if(row == len(board)):
                res.append(turnToStr(board))
                return
            nCol = len(board[row])
            for col in range(nCol):
                if(not isValid(board, row, col)):
                    continue
                board[row][col] = 'Q'
                recall(board, row + 1)
                board[row][col] = '.'
        recall(board, 0)
        return res
    """
    111. 二叉树的最小深度
    给定一个二叉树，找出其最小深度。
    最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
    说明：叶子节点是指没有子节点的节点。
    """
    def minDepth(self, root: TreeNode) -> int:
        """
        BFS
        :param root:
        :return:
        """
        res = 0
        if(not root):
            return res
        q = Queue()
        q.put(root)
        l = Queue()
        while(not q.empty()):
            node = q.get()
            # print(q.empty())
            if(not node.left and not node.right):
                res += 1
                break
            if(node.left):
                l.put(node.left)
            if(node.right):
                l.put(node.right)

            if(q.empty()):
                res += 1
                # print(res)
                if(not l.empty()):
                    while(not l.empty()):
                        it = l.get()
                        q.put(it)
                else:
                    break
        return res

    def minDepthOthers(self, root: TreeNode) -> int:
        res = 0
        if (not root):
            return res
        q = Queue()
        q.put(root)
        res += 1

        while(not q.empty()):
            nSize = q.qsize()
            for i in range(nSize):
                node = q.get()
                if(not node.left and not node.right):
                    return res
                if(node.left):
                    q.put(node.left)
                if(node.right):
                    q.put(node.right)
            res += 1
        return res

    """
    752. 打开转盘锁
    你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：
    例如把 '9' 变为  '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。
    锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。
    列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。
    字符串 target 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能解锁，返回 -1。
    """
    def openLock(self, deadends: List[str], target: str) -> int:
        def plusOne(s: str, i: int):
            l = list(s)
            n = int(l[i])
            n += 1
            if(n > 9):
                n = 0
            l[i] = str(n)
            return ''.join(l)
        def minusOne(s: str, i: int):
            l = list(s)
            n = int(l[i])
            n -= 1
            if (n < 0):
                n = 9
            l[i] = str(n)
            return ''.join(l)

        q = Queue()
        q.put('0000')
        memo = set()
        memo.add('0000')
        res = 0
        while(not q.empty()):
            nSize = q.qsize()
            for i in range(nSize):
                s = q.get()
                if(s in deadends):
                    continue
                if(s == target):
                    return res
                n = len(s)
                for i in range(n):
                    plus = plusOne(s, i)
                    minus = minusOne(s, i)
                    if(plus not in deadends and plus not in memo):
                        q.put(plus)
                        memo.add(plus)
                    if(minus not in deadends and minus not in memo):
                        q.put(minus)
                        memo.add(minus)
            res += 1
        return -1

    """
    704. 二分查找
    给定一个  n  个元素有序的（升序）整型数组  nums 和一个目标值  target   ，写一个函数搜索  nums  中的 target，
    如果目标值存在返回下标，否则返回 -1。
    """
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if(n == 0):
            return -1
        left = 0
        right = n-1

        while(left <= right):
            middle = left + (right - left)//2
            it = nums[middle]
            if(it == target):
                return middle;
            elif(it > target):
                right = middle - 1
            else:
                left = middle + 1
        return -1
    """
    34. 在排序数组中查找元素的第一个和最后一个位置
    给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
    如果数组中不存在目标值 target，返回 [-1, -1]。

    进阶：
    你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？
    """
    def searchLeftBound(self, nums: List[int], target):
        n = len(nums)
        if (n == 0):
            return -1
        left = 0
        right = n

        while (left < right):
            middle = left + (right - left) // 2
            it = nums[middle]
            if (it == target):
                right = middle
            elif (it > target):
                right = middle
            else:
                left = middle + 1
        if (left >= n):
            return -1
        return -1 if (nums[left] != target) else left
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        idx = self.searchLeftBound(nums, target)

        if(idx == -1):
            return [-1, -1]
        else:
            right = idx
            for i in range(idx+1, n):
                if(nums[i] == target):
                    right = i
            return [idx, right]

    """
    76. 最小覆盖子串
    给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
    注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。
    """
    def minWindow(self, s: str, t: str) -> str:
        """
        滑动窗口
        :param s:
        :param t:
        :return:
        """
        n = len(s)
        if(n == 0):
            return ''
        need = {}
        window = {}
        for c in t:
            if(c in need):
                need[c] += 1
            else:
                need[c] = 1
        left = 0
        right = 0
        valid = 0
        start = 0
        minLen = sys.maxsize
        while(right < n):
            c = s[right]
            right += 1
            if(c in need):
                if(c in window):
                    window[c] += 1
                else:
                    window[c] = 1
                if(window[c] == need[c]):
                    valid += 1
            # 满足条件，开始收缩窗口
            while(valid == len(need)):
                # 更新结果
                if(right - left < minLen):
                    start = left
                    minLen = right - left
                w = s[left]
                left += 1
                if(w in need):
                    if(w in window):
                        if(need[w] == window[w]):
                            valid -= 1
                        window[w] -= 1
        return '' if(minLen == sys.maxsize) else s[start: start + minLen]
    """
    567. 字符串的排列
    给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的排列。
    换句话说，第一个字符串的排列之一是第二个字符串的子串。
    """
    def checkInclusion(self, s1: str, s2: str) -> bool:
        """
        滑动窗口
        :param s:
        :param t:
        :return:
        """
        n = len(s2)
        m = len(s1)
        if (n == 0):
            return ''
        need = {}
        window = {}
        # 目标字符串的映射
        for c in s1:
            if (c in need):
                need[c] += 1
            else:
                need[c] = 1
        left = 0
        right = 0
        valid = 0
        while (right < n):
            c = s2[right]
            right += 1
            # 移动右指针，将字符加入窗口
            if (c in need):
                if (c in window):
                    window[c] += 1
                else:
                    window[c] = 1
                if (window[c] == need[c]):
                    valid += 1
            # 符合条件
            while(valid == len(need)):
                # 找到子串与s1的排列一致
                if (right - left == m):
                    return True
                else:
                    # 移动左指针，收缩窗口
                    w = s2[left]
                    left += 1
                    if (w in need):
                        if (w in window):
                            if (need[w] == window[w]):
                                valid -= 1
                            window[w] -= 1
        return False
    """
    438. 找到字符串中所有字母异位词
    给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。
    字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。
    
    说明：
    字母异位词指字母相同，但排列不同的字符串。
    不考虑答案输出的顺序。
    """
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if (not s):
            return []

        n = len(s)
        m = len(p)

        need = {}
        window = {}

        for c in p:
            if (c in need):
                need[c] += 1
            else:
                need[c] = 1
        left = 0
        right = 0
        valid = 0
        res = []
        while (right < n):
            c = s[right]
            right += 1
            if (c in need):
                if (c in window):
                    window[c] += 1
                else:
                    window[c] = 1
                if (window[c] == need[c]):
                    valid += 1

            while (valid == len(need)):
                if (right - left == m):
                    res.append(left)

                w = s[left]
                left += 1
                if (w in need):
                    if (w in window):
                        if (window[w] == need[w]):
                            valid -= 1
                        window[w] -= 1
        return res
    """
    3. 无重复字符的最长子串
    给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
    """
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        if (n == 0):
            return 0
        window = set()
        left = 0
        right = 0
        res = 0
        while (right < n):
            c = s[right]
            right += 1
            if (c in window):
                while (left < right):
                    w = s[left]
                    window.remove(w)
                    left += 1
                    if (w == c):
                        window.add(c)
                        break
            else:
                window.add(c)
                l = len(window)
                res = max(res, l)
        return res
    """
    121. 买卖股票的最佳时机
    给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
    你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
    返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
    """
    def maxProfit(self, prices: List[int]) -> int:
        """
        动态规划
        :param prices:
        :return:
        """
        n = len(prices)
        if(n == 0):
            return 0
        dp = [[0,0] for _ in range(n)]
        # 边界条件, 0表示未持有股票，1表示当前持有股票
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        # 只能进行一次买入卖出
        for i in range(1, n):
            # 当前未持有股票，昨天未持有，或者昨天持有但是今天卖出
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            # 当前持有股票，昨天就持有，或者昨天未持有但是今天第一次买入 0 - prices[i]
            dp[i][1] = max(dp[i-1][1], - prices[i])
        return dp[n-1][0]
    """
    122. 买卖股票的最佳时机 II
    给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
    设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    """
    def maxProfitII(self, prices: List[int]) -> int:
        n = len(prices)
        if(n == 0):
            return 0
        dp = [[0,0 ] for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        # 注意：与121题不同的是，可以进行无数次买入卖出
        for i in range(1, n):
            # 当前未持有股票，昨天未持有，或者昨天持有但是今天卖出
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            # 当前持有股票，昨天就持有，或者昨天未持有但是今天第一次买入 0 - prices[i]
            dp[i][1] = max(dp[i - 1][1], dp[i-1][0] - prices[i])
        return dp[n - 1][0]
    """
    123. 买卖股票的最佳时机 III
    给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
    设计一个算法来计算你所能获取的最大利润。你最多可以完成  两笔  交易。
    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    """
    def maxProfitIII(self, prices: List[int]) -> int:
        n = len(prices)
        if(n == 0):
            return 0
        dp = [[[0, 0] for i in range(3)] for j in range(n)]
        # 边界条件
        # 允许的交易次数为 0时，持有股票时，收益为 负无穷, 未持有股票时的收益为 0
        for i in range(n):
            dp[i][0][0] = 0
            dp[i][0][1] = float('-inf')
        # 第 0 天，持有股票时，收益为 -prices[0], 未持有股票时的收益为 0
        for j in range(1,3):
            dp[0][j][0] = 0
            dp[0][j][1] = -prices[0]

        for i in range(1, n):
            for j in range(1, 3):
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i])
                # 当天持有股票的收益，等于昨日就持有股票 和 昨日为持有股票并买入股票之间的较大值
                dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i])

        return dp[n-1][2][0]

    """
    188. 买卖股票的最佳时机 IV
    给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
    设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    """
    def maxProfitIV(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        if (n == 0):
            return 0
        dp = [[[0, 0] for i in range(k + 1)] for j in range(n)]
        # 边界条件
        # 允许的交易次数为 0时，持有股票时，收益为 负无穷, 未持有股票时的收益为 0
        for i in range(n):
            dp[i][0][0] = 0
            dp[i][0][1] = float('-inf')
        # 第 0 天，持有股票时，收益为 -prices[0], 未持有股票时的收益为 0
        for j in range(1, k + 1):
            dp[0][j][0] = 0
            dp[0][j][1] = -prices[0]

        for i in range(1, n):
            for j in range(1, k + 1):
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i])
                # 当天持有股票的收益，等于昨日就持有股票 和 昨日为持有股票并买入股票之间的较大值
                dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i])

        return dp[n - 1][k][0]
    """
    309. 最佳买卖股票时机含冷冻期
    给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。

    设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
    1.你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    2.卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
    """
    def maxProfitVI(self, prices: List[int]) -> int:
        n = len(prices)
        if (n == 0):
            return 0
        dp = [[0, 0] for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        # 注意：与121题不同的是，可以进行无数次买入卖出
        for i in range(1, n):
            # 当前未持有股票，昨天未持有，或者昨天持有但是今天卖出
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            # 当前持有股票，昨天就持有，或者前天（冷冻期）未持有但是今天买入 - prices[i]
            dp[i][1] = max(dp[i - 1][1], dp[i - 2][0] - prices[i])
        return dp[n - 1][0]
    """
    714. 买卖股票的最佳时机含手续费
    给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。

    你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
    返回获得利润的最大值。
    注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
    """
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        if (n == 0):
            return 0
        dp = [[0, 0] for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        # 注意：与121题不同的是，可以进行无数次买入卖出
        for i in range(1, n):
            # 当前未持有股票，昨天未持有，或者昨天持有但是今天卖出
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
            # 当前持有股票，昨天就持有，或者前天（冷冻期）未持有但是今天买入 - prices[i]
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[n - 1][0]
    """
    198. 打家劫舍
    你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
    如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

    给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
    """
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if(n == 0):
            return 0
        # dp 表示前 n 间住房所能偷盗的最大金额
        dp = [0 for _ in range(n + 1)]
        # 边界条件，没有住房可偷为 0，只有一间住房时，只能偷这个房间的金额 nums[0]
        dp[0] = 0
        dp[1] = nums[0]
        for i in range(2, n + 1):
            # dp[i] 前 i 间住房所能偷盗的最大金额可以表示为：
            # 前 i-1 间住房所能偷盗的最大金额
            # 前 i-2 间住房所能偷盗的最大金额 + 第 i 间住房的金额
            # 两者之间的最大值
            dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])
        return dp[n]

    """
    213. 打家劫舍 II
    你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，
    这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，
    系统会自动报警 。

    给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，能够偷窃到的最高金额。
    """
    def robII(self, nums: List[int]) -> int:
        def robRange(nums):
            n = len(nums)
            if (n == 0):
                return 0
            # dp 表示前 n 间住房所能偷盗的最大金额
            dp = [0 for _ in range(n + 1)]
            # 边界条件，没有住房可偷为 0，只有一间住房时，只能偷这个房间的金额 nums[0]
            dp[0] = 0
            dp[1] = nums[0]
            for i in range(2, n + 1):
                # dp[i] 前 i 间住房所能偷盗的最大金额可以表示为：
                # 前 i-1 间住房所能偷盗的最大金额
                # 前 i-2 间住房所能偷盗的最大金额 + 第 i 间住房的金额
                # 两者之间的最大值
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
            return dp[n]
        n = len(nums)
        if(n == 1):
            return nums[0]
        # 最后一间住房与第一件住房不能同时被抢：
        # 抢最后一间住房 或者 抢第一间住房
        res = max(robRange(nums[0:n-1]), robRange(nums[1:]))
        return res
    """
    337. 打家劫舍 III
    在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 
    除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 
    如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

    计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
    """
    def robIII(self, root: TreeNode) -> int:
        if(not root):
            return 0
        if(root in self.memo):
            return self.memo[root]

        steal = root.val + ((self.robIII(root.left.left) + self.robIII(root.left.right)) if(root.left) else 0) + ((self.robIII(root.right.left) + self.robIII(root.right.right)) if(root.right) else 0)
        not_steal = self.robIII(root.left) + self.robIII(root.right)

        res = max(steal, not_steal)

        self.memo[root] = res

        return res

    """
    1288. 删除被覆盖区间
    给你一个区间列表，请你删除列表中被其他区间所覆盖的区间。
    只有当 c <= a 且 b <= d 时，我们才认为区间 [a,b) 被区间 [c,d) 覆盖。
    在完成所有删除操作后，请你返回列表中剩余区间的数目。
    """
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        # 排序，按起点升序排列，起点相同，终点降序排列
        def compareRule(a, b):
            if (a[0] == b[0]):
                return b[1] - a[1]
            return a[0] - b[0]
        intervals.sort(key=functools.cmp_to_key(compareRule))
        n = len(intervals)
        # 三种情况，覆盖，相交，不相交
        start = intervals[0][0]
        end = intervals[0][1]
        ret = 0
        for idx in range(1, n):
            s = intervals[idx][0]
            e = intervals[idx][1]
            # 覆盖
            if(start <= s and end >= e):
                ret += 1
            # 相交
            if(end >= s and end <= e):
                end = e
            # 不相交
            if(end <= s):
                start = s
                end = e
        return n - ret
    """
    56. 合并区间
    以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，
    并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。
    """
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 排序，按起点升序排列，起点相同，终点降序排列
        def compareRule(a, b):
            if (a[0] == b[0]):
                return b[1] - a[1]
            return a[0] - b[0]

        intervals.sort(key=functools.cmp_to_key(compareRule))
        n = len(intervals)
        # 三种情况，覆盖，相交，不相交
        start = intervals[0][0]
        end = intervals[0][1]
        ret = [[start, end]]
        for idx in range(1, n):
            s = intervals[idx][0]
            e = intervals[idx][1]
            # 覆盖
            # if (start <= s and end >= e):
            # ret.append([start, end])
            # 相交
            if (end >= s and end <= e):
                end = e
                ret[-1][1] = end
            # 不相交
            if (end < s):
                start = s
                end = e
                ret.append([start, end])
        return ret
    """
    986. 区间列表的交集
    给定两个由一些 闭区间 组成的列表，firstList 和 secondList ，其中 firstList[i] = [starti, endi] 
    而 secondList[j] = [startj, endj] 。每个区间列表都是成对 不相交 的，并且 已经排序 。
    返回这 两个区间列表的交集 。
    形式上，闭区间 [a, b]（其中 a <= b）表示实数 x 的集合，而 a <= x <= b 。
    两个闭区间的 交集 是一组实数，要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3]。
    """
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        n = len(firstList)
        m = len(secondList)
        if(not n or not m):
            return []

        res = []
        i, j = 0, 0
        while(i < n and j < m):
            a1, a2 = firstList[i][0], firstList[i][1]
            b1, b2 = secondList[j][0], secondList[j][1]
            # 有交集的情况, !(a2 < b1 or b2 < a1)
            if(b1 <= a2 and a1 <= b2):
                c1 = max(a1, b1)
                c2 = min(a2, b2)
                res.append([c1, c2])
            if(b2 < a2):
                j += 1
            else:
                i += 1
        return res

    """
    15.三数之和（中等）
    给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？
    请你找出所有和为 0 且不重复的三元组。
    注意：答案中不可以包含重复的三元组。
    """
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        def twoSum(nums: List[int], target: int):
            res = []
            n = len(nums)
            if(n == 0):
                return res
            # nums.sort(reverse=False)
            lo = 0
            hi = n-1
            while(lo < hi):
                left = nums[lo]
                right = nums[hi]
                s = left + right

                if(s < target):
                    while(lo < hi and nums[lo] == left):
                        lo += 1
                elif(s > target):
                    while(lo < hi and nums[hi] == right):
                        hi -= 1
                else:
                    res.append([left, right])
                    while(lo < hi and nums[lo] == left):
                        lo += 1
                    while(lo < hi and nums[hi] == right):
                        hi -= 1
            return res
        ret = []
        n = len(nums)
        if (n == 0):
            return ret
        nums.sort(reverse=False)
        i = 0
        while (i < n):
            it = nums[i]
            binary = twoSum(nums[i + 1:], 0 - it)
            for e in binary:
                e.append(it)
                ret.append(e)
            i += 1
            while (i < n - 1 and it == nums[i]):
                i += 1
        return ret

    """
    1. 两数之和
    给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 的那 两个 整数，并返回它们的数组下标。
    你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
    你可以按任意顺序返回答案。
    """
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hash = dict()
        res = []
        n = len(nums)
        if (n == 0):
            return res
        for i in range(n):
            hash[target - nums[i]] = i
        for i in range(n):
            item = nums[i]
            if (item in hash):
                another = hash[item]
                if (another not in res and another != i):
                    res.append(i)
                    res.append(another)
        return res

    """
    18. 四数之和
    给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，
    使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。

    注意：答案中不可以包含重复的四元组。
    """
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        def twoSum(nums: List[int], target: int) -> List[List[int]]:
            res = []
            n = len(nums)
            if (n == 0):
                return res
            # nums.sort(reverse=False)
            lo = 0
            hi = n - 1
            while (lo < hi):
                left = nums[lo]
                right = nums[hi]
                s = left + right

                if (s < target):
                    while (lo < hi and nums[lo] == left):
                        lo += 1
                elif (s > target):
                    while (lo < hi and nums[hi] == right):
                        hi -= 1
                else:
                    res.append([left, right])
                    while (lo < hi and nums[lo] == left):
                        lo += 1
                    while (lo < hi and nums[hi] == right):
                        hi -= 1
            return res
        def threeSum(nums: List[int], target: int) -> List[List[int]]:
            ret = []
            n = len(nums)
            if (n == 0):
                return ret
            # nums.sort(reverse=False)
            i = 0
            while (i < n):
                it = nums[i]
                binary = twoSum(nums[i + 1:], target - it)
                for e in binary:
                    e.append(it)
                    ret.append(e)
                i += 1
                while (i < n - 1 and it == nums[i]):
                    i += 1
            return ret
        res = []
        n = len(nums)
        if(n == 0):
            return res
        nums.sort(reverse=False)
        i = 0
        while(i < n):
            it = nums[i]
            triplet = threeSum(nums[i+1:], target-it)
            for t in triplet:
                t.append(it)
                res.append(t)
            while(i < n and it == nums[i]):
                i += 1
        return res
    """
    226. 翻转二叉树
    翻转一棵二叉树。
    """
    def invertTree(self, root: TreeNode) -> TreeNode:
        if(not root):
            return
        tmp = root.right
        root.right = root.left
        root.left = tmp
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
    """
    114. 二叉树展开为链表
    给你二叉树的根结点 root ，请你将它展开为一个单链表：
    
    展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
    展开后的单链表应该与二叉树 先序遍历 顺序相同。
    """
    def flatten(self, root: TreeNode) -> None:
        """
        后续遍历二叉树
        """
        if(not root):
            return
        self.flatten(root.left)
        self.flatten(root.right)

        left = root.left
        right = root.right
        root.left = None
        root.right = left

        p = root
        while(p.right):
            p = p.right
        p.right = right

    """
    116. 填充每个节点的下一个右侧节点指针
    给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。
    填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
    初始状态下，所有 next 指针都被设置为 NULL。
    """
    def connect(self, root: 'Node') -> 'Node':
        if (not root):
            return
        if (root.left):
            root.left.next = root.right
        if (root.right):
            root.right.next = root.next.left if (root.next) else None
        self.connect(root.left)
        self.connect(root.right)
        return root
    """
    416. 分割等和子集
    给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

    注意:
    每个数组中的元素不会超过 100
    数组的大小不会超过 200
    """
    def canPartition(self, nums: List[int]) -> bool:
        s = sum(nums)
        if(s % 2 != 0):
            return False
        half = s // 2
        n = len(nums)
        # 变成了背包问题
        dp = [[False for i in range(half + 1)] for j in range(n + 1)]
        for i in range(n+1):
            dp[i][0] = True
        for i in range(1, n + 1):
            for j in range(1, half + 1):
                if(nums[i-1] <= j):
                    # 满足条件时，可以选择装入或者不装入
                    dp[i][j] = dp[i-1][j-nums[i-1]] or dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[n][half] == 1
    """
    92. 反转链表 II
    反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
    """
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        successor = None
        # 反转链表的前n个节点
        def reverseN(head: ListNode, N: int):
            nonlocal successor
            if(N == 1):
                successor = head.next
                return head
            last = reverseN(head.next, N-1)
            head.next.next = head
            head.next = successor
            return last
        if(left == 1):
            return reverseN(head, right)
        head.next = self.reverseBetween(head.next, left-1, right-1)
        return head
    """
    25. K 个一组翻转链表
    给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

    k 是一个正整数，它的值小于或等于链表的长度。

    如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

    进阶：
    你可以设计一个只使用常数额外空间的算法来解决此问题吗？
    你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
    """
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        def reverseBetwen(a: ListNode, b: ListNode) -> ListNode:
            pre = None
            cur = a
            nxt = a
            while(cur != b):
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt
            return pre
        if(not head):
            return None
        a = head
        b = head
        for i in range(k):
            if(not b):
                return head
            b = b.next
        newHead = reverseBetwen(a, b)
        a.next = self.reverseKGroup(b, k)
        return newHead
    """
    234. 回文链表
    请判断一个链表是否为回文链表。
    """
    def isPalindrome(self, head: ListNode) -> bool:
        """
        双指针，后序遍历，递归
        :param head:
        :return:
        """
        left = head
        def traverse(head: ListNode) -> bool:
            """
            单链表的后序遍历
            :param head:
            :return:
            """
            nonlocal left
            if(not head):
                return True
            res = traverse(head.next)
            res = res and (left.val == head.val)
            left = left.next
            return res
        return traverse(head)
    """
    654. 最大二叉树
    给定一个不含重复元素的整数数组 nums 。一个以此数组直接递归构建的 最大二叉树 定义如下：

    二叉树的根是数组 nums 中的最大元素。
    左子树是通过数组中 最大值左边部分 递归构造出的最大二叉树。
    右子树是通过数组中 最大值右边部分 递归构造出的最大二叉树。
    返回有给定数组 nums 构建的 最大二叉树 。
    """
    def getMax(self, nums: List[int]):
        res = None
        resIdx = None
        for idx, i in enumerate(nums):
            if (res is None):
                res = i
                resIdx = idx
            elif (res < i):
                res = i
                resIdx = idx
        return (res, resIdx)
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        if(not nums):
            return None
        m = max(nums)
        idx = nums.index(m)
        root = TreeNode(val=m)
        left = self.constructMaximumBinaryTree(nums[:idx])
        right = self.constructMaximumBinaryTree(nums[idx+1:])
        root.left = left
        root.right = right
        return root
    """
    105. 从前序与中序遍历序列构造二叉树
    根据一棵树的前序遍历与中序遍历构造二叉树。
    
    注意:
    你可以假设树中没有重复的元素。
    例如，给出

    前序遍历 preorder = [3,9,20,15,7]
    中序遍历 inorder = [9,3,15,20,7]
    返回如下的二叉树：
    
        3
       / \
      9  20
        /  \
       15   7
    """
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def build(preorder: List[int] , inorder: List[int]) -> TreeNode:
            if(not preorder or not inorder):
                return None
            rootVal = preorder[0]
            inIdx = inorder.index(rootVal)
            root = TreeNode(rootVal)
            root.left = build(preorder[1: inIdx+1], inorder[: inIdx])
            root.right = build(preorder[inIdx+1:], inorder[inIdx + 1:])
            return root
        return build(preorder, inorder)
    """
    106. 从中序与后序遍历序列构造二叉树
    根据一棵树的中序遍历与后序遍历构造二叉树。

    注意:
    你可以假设树中没有重复的元素。
    
    例如，给出
    
    中序遍历 inorder = [9,3,15,20,7]
    后序遍历 postorder = [9,15,7,20,3]
    返回如下的二叉树：
    
        3
       / \
      9  20
        /  \
       15   7
    """
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        def build(inorder: List[int], postorder: List[int]) -> TreeNode:
            if(not inorder or not postorder):
                return None
            rootVal = postorder[-1]
            inIdx = inorder.index(rootVal)
            root = TreeNode(rootVal)
            root.left = build(inorder[:inIdx], postorder[:inIdx])
            root.right = build(inorder[inIdx+1:], postorder[inIdx:-1])
            return root
        return build(inorder, postorder)
    """
    652. 寻找重复的子树
    给定一棵二叉树，返回所有重复的子树。对于同一类的重复子树，你只需要返回其中任意一棵的根结点即可。
    两棵树重复是指它们具有相同的结构以及相同的结点值。
    
    示例 1：
    
            1
           / \
          2   3
         /   / \
        4   2   4
           /
          4
    下面是两个重复的子树：
    
          2
         /
        4
    和
    
        4
    """
    def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
        resMap = dict()
        res = []
        def traverse(root: TreeNode) -> str:
            if(not root):
                return '#'
            left = traverse(root.left)
            right = traverse(root.right)
            rootStr = left + ',' + right + ',' + str(root.val)
            if(rootStr in resMap):
                if(resMap[rootStr] == 1):
                    res.append(root)
                resMap[rootStr] = resMap[rootStr] + 1
            else:
                 resMap[rootStr] = 1
            return rootStr
        traverse(root)
        return res
    """
    230. 二叉搜索树中第K小的元素
    给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。
    """
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        res = []
        def traverse(root: TreeNode):
            if(not root):
                return
            traverse(root.left)
            if(len(res) == k):
                return
            res.append(root.val)
            traverse(root.right)
        traverse(root)
        return res[-1]
    """
    538. 把二叉搜索树转换为累加树
    给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。

    提醒一下，二叉搜索树满足下列约束条件：
    
    节点的左子树仅包含键 小于 节点键的节点。
    节点的右子树仅包含键 大于 节点键的节点。
    左右子树也必须是二叉搜索树。
    注意：本题和 1038: https://leetcode-cn.com/problems/binary-search-tree-to-greater-sum-tree/ 相同
    """
    def convertBST(self, root: TreeNode) -> TreeNode:
        sum = 0

        def traverseSum(root: TreeNode):
            nonlocal sum
            if (not root):
                return
            traverseSum(root.right)
            sum = sum + root.val
            root.val = sum
            traverseSum(root.left)

        traverseSum(root)
        return root
    """
    450. 删除二叉搜索树中的节点
    给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

    一般来说，删除节点可分为两个步骤：
    
    首先找到需要删除的节点；
    如果找到了，删除它。
    说明： 要求算法时间复杂度为 O(h)，h 为树的高度。
    """
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        def getMin(root: TreeNode):
            if(not root):
                return None
            while(root.left):
                root = root.left
            return root
        if(not root):
            return None
        if(root.val == key):
            if(not root.left and not root.right):
                return None
            elif(not root.right):
                return root.left
            elif(not root.left):
                return root.right
            minNode = getMin(root.right)
            root.val = minNode.val
            root.right = self.deleteNode(root.right, minNode.val)
        elif(root.val > key):
            root.left = self.deleteNode(root.left, key)
        elif(root.val < key):
            root.right = self.deleteNode(root.right, key)
        return root
    """
    701. 二叉搜索树中的插入操作
    给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。

    注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 任意有效的结果 。
    """
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if (not root):
            return TreeNode(val)
        elif (root.val < val):
            root.right = self.insertIntoBST(root.right, val)
        elif (root.val > val):
            root.left = self.insertIntoBST(root.left, val)
        return root
    """
    700. 二叉搜索树中的搜索
    给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。
    """
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if (not root):
            return None
        elif (root.val == val):
            return root
        elif (root.val < val):
            return self.searchBST(root.right, val)
        elif (root.val > val):
            return self.searchBST(root.left, val)
    """
    98. 验证二叉搜索树
    给定一个二叉树，判断其是否是一个有效的二叉搜索树。

    假设一个二叉搜索树具有如下特征：
    
    节点的左子树只包含小于当前节点的数。
    节点的右子树只包含大于当前节点的数。
    所有左子树和右子树自身必须也是二叉搜索树。
    """
    def isValidBST(self, root: TreeNode) -> bool:
        def isValid(root: TreeNode, min: TreeNode, max: TreeNode):
            if(not root):
                return True
            elif(max and root.val <= max.val):
                return False
            elif(min and root.val >= min.val):
                return False
            return isValid(root.left, root, max) and isValid(root.right, min, root)
        return isValid(root, None, None)
    
    """
    236. 二叉树的最近公共祖先
    给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

    百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，
    最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
    """
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if(not root or root == p or root == q):
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if(left and right):
            return root
        elif(not left and not right):
            return None
        else:
            return left if(left) else right
    """
    222. 完全二叉树的节点个数
    给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。

    完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，
    并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。
    """
    def countNodes(self, root: TreeNode) -> int:
        size = 0

        def nodeSize(root: TreeNode):
            nonlocal size
            if (not root):
                return
            size += 1
            nodeSize(root.left)
            nodeSize(root.right)

        nodeSize(root)
        return size
    """
    130. 被围绕的区域
    给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
    """
    def solve(self, board: List[List[str]]) -> None:
        """
        二维坐标[x][y]转换为一维坐标 x * n + y.
        """
        if(not board):
            return
        m = len(board)
        n = len(board[0])
        uf = UnionFind(m * n + 1)
        dummy = m * n
        # 首列与末列中的 O 与 dummy相连接
        for i in range(m):
            if(board[i][0] == 'O'):
                uf.union(dummy, i * n)
            if(board[i][n-1] == 'O'):
                uf.union(dummy, i * n + n -1)
        # 首行与末行中的 O 与 dummy相连接
        for j in range(n):
            if(board[0][j] == 'O'):
                uf.union(dummy, j)
            if(board[m-1][j] == 'O'):
                uf.union(dummy, (m-1) * n + j)

        # 方向数组
        d = [[1, 0], [0, 1], [0, -1], [-1, 0]]
        for i in range(1, m-1):
            for j in range(1, n-1):
                if(board[i][j] == 'O'):
                    # 将周围的 O 与 当前的 O 连接
                    for k in range(4):
                        x = i + d[k][0]
                        y = j + d[k][1]
                        if(board[x][y] == 'O'):
                            uf.union(x * n + y, i * n + j)

        # 所有不与 dummy 连接的 O 替换为 X
        for i in range(1, m-1):
            for j in range(1, n-1):
                if(board[i][j] == 'O'):
                    if(not uf.connected(dummy, i * n + j)):
                        board[i][j] = 'X'
    """
    990. 等式方程的可满足性
    给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations[i] 的长度为 4，
    并采用两种不同的形式之一："a==b" 或 "a!=b"。在这里，a 和 b 是小写字母（不一定不同），表示单字母变量名。
    只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，否则返回 false。
    """
    def equationsPossible(self, equations: List[str]) -> bool:
        if(not equations):
            return
        # == 关系表示连通性
        a = ord('a')
        uf = UnionFind(26)
        for equation in equations:
            if(equation[1] == '='):
                uf.union(ord(equation[0])-a, ord(equation[3])-a)
        # != 关系表示不连通
        for equation in equations:
            if(equation[1] == '!'):
                if(uf.connected(ord(equation[0])-a, ord(equation[3])-a)):
                    return False
        return True
    """
    496. 下一个更大元素 I
    给你两个 没有重复元素 的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。
    请你找出 nums1 中每个元素在 nums2 中的下一个比其大的值。
    nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出 -1 。
    """
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        res = []
        resMap = {}
        if(not nums1 or not nums2):
            return res
        len2 = len(nums2)
        for idx in range(len2-1, -1, -1):
            while(stack and stack[-1] <= nums2[idx]):
                stack.pop()
            resMap[nums2[idx]] = stack[-1] if(stack) else -1
            stack.append(nums2[idx])

        for idx, i in enumerate(nums1):
            res.append(resMap[i])
        return res