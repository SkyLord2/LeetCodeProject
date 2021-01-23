# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


"""
字符串匹配 KMP 算法
通过 确定有限状态自动机(deterministic finit automaton, DFA) 来实现, 详细的理论见编译原理 
"""
class KMP:
    def __init__(self, pat: str):
        self.pat = pat
        # 构建自动机的状态转移矩阵
        self.length = len(pat)
        self.dp = [[0 for _ in range(256)] for i in range(self.length)]
        if (self.length > 0):
            # 当前状态的前一个状态
            X = 0
            # 匹配到模式串的第一个字符时，状态从 0 转移到 1
            self.dp[0][ord(pat[0])] = 1
            # 状态从 1 开始
            for i in range(1, self.length):
                for j in range(256):
                    # 匹配，状态前进
                    if(ord(self.pat[i]) == j):
                        self.dp[i][j] = i+1
                    # 不匹配，状态适当后退
                    else:
                        self.dp[i][j] = self.dp[X][j]
                X = self.dp[X][ord(self.pat[i])]

    def search(self, txt: str):
        if (self.length == 0):
            return 0
        else:
            N = len(txt)
            j = 0
            for i in range(N):
                # 从状态矩阵中获取 下一个 状态
                j = self.dp[j][ord(txt[i])]
                # 到达终止状态
                if(j == self.length):
                    return i - self.length + 1
            return -1

"""
460. LFU 缓存
请你为 最不经常使用（LFU）缓存算法设计并实现数据结构。

实现 LFUCache 类：
LFUCache(int capacity) - 用数据结构的容量 capacity 初始化对象
int get(int key) - 如果键存在于缓存中，则获取键的值，否则返回 -1。
void put(int key, int value) - 如果键已存在，则变更其值；如果键不存在，请插入键值对。当缓存达到其容量时，则应该在插入新项之前，使最不经常使用的项无效。在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，应该去除 最久未使用 的键。

注意「项的使用次数」就是自插入该项以来对其调用 get 和 put 函数的次数之和。使用次数会在对应项被移除后置为 0 。

进阶：   
你是否可以在 O(1) 时间复杂度内执行两项操作？
"""
class LinkedHashSet:
    def __init__(self):
        self.l = []
        self.s = set()
    def add(self, val):
        if(val not in self.s):
            self.l.append(val)
            self.s.add(val)
        return self
    def get(self, idx):
        if(idx >= len(self.l)):
            return None
        else:
            return self.l[idx]

    def remove(self, val):
        if(val in self.s):
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
        if(key in self.keyToVal):
            self.increaseFreq(key)
            return self.keyToVal[key]
        else:
            return -1
    def increaseFreq(self, key):
        if(key in self.keyToFreq):
            freq = self.keyToFreq[key]
            oringinKeys = self.freqToKeys[freq]
            oringinKeys.remove(key)
            freq += 1
            if(oringinKeys.size() == 0):
                del self.freqToKeys[freq]
                if(freq == self.minFreq):
                    self.minFreq = freq

            if(freq in self.freqToKeys):
                self.freqToKeys[freq].add(key)
            else:
                self.freqToKeys[freq] = LinkedHashSet().add(key)

            self.keyToFreq[key] = freq
        else:
            self.keyToFreq[key] = 1
            if(1 in self.freqToKeys):
                self.freqToKeys[1].add(key)
            else:
                self.freqToKeys[1] = LinkedHashSet().add(key)
            self.minFreq = 1

    def removeMinFreq(self):
        keys = self.freqToKeys[self.minFreq]
        oldKey = keys.get(0)
        keys.remove(oldKey)
        if(keys.size() == 0):
            del self.freqToKeys[self.minFreq]

        del self.keyToFreq[oldKey]

        del self.keyToVal[oldKey]

    def put(self, key: int, value: int) -> None:
        if(self.capacity <= 0):
            return
        if(key in self.keyToVal):
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
并查集Union-Find, 通过数组模拟一个森林
1、自反性：节点p和p是连通的。
2、对称性：如果节点p和q连通，那么q和p也连通。
3、传递性：如果节点p和q连通，q和r连通，那么p和r也连通。
"""
class UF:
    def __init__(self, count):
        # 记录连通分量
        self.count = count
        # 记录节点的父节点
        self.parent = [i for i in range(count)]
        # 记录树的重量
        self.size = [1 for _ in range(count)]

    def find(self, x):
        while(x != self.parent[x]):
            # 路径压缩
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, p, q):
        """
        将p和q链接
        :param p:
        :param q:
        :return:
        """
        rootP = self.find(p)
        rootQ = self.find(q)
        if(rootP == rootQ):
            return

        if(self.size[rootP] > self.size[rootQ]):
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        self.count += 1
    def connected(self, p, q):
        """
        p和q是否连通
        :param p:
        :param q:
        :return:
        """
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ
    def count(self):
        """
        图中有多少个连通分量
        :return:
        """
        return self.count

"""
855. 考场就座
在考场里，一排有 N 个座位，分别编号为 0, 1, 2, ..., N-1 。
当学生进入考场后，他必须坐在能够使他与离他最近的人之间的距离达到最大化的座位上。如果有多个这样的座位，他会坐在编号最小的座位上。
(另外，如果考场里没有人，那么学生就坐在 0 号座位上。)
返回 ExamRoom(int N) 类，它有两个公开的函数：其中，函数 ExamRoom.seat() 会返回一个 int （整型数据），
代表学生坐的位置；函数 ExamRoom.leave(int p) 代表坐在座位 p 上的学生现在离开了考场。每次调用 ExamRoom.leave(p) 时都保证有学生坐在座位 p 上。
"""
class ExamRoom:

    def __init__(self, N):
        self.N = N
        # 记录学生的位置
        self.students = []

    def seat(self):
        # 还没有学生，直接坐在第一个位置
        if not self.students:
            student = 0
        else:
            # 有学生，要比较相邻学生之间的位置
            dist, student = self.students[0], 0
            for i, s in enumerate(self.students):
                if i:
                    prev = self.students[i - 1]
                    # 与前一个学生之间的中间的位置（小索引优先）
                    d = int((s - prev) / 2)
                    # 记录最大的间隔，以及插入的位置
                    if d > dist:
                        dist, student = d, prev + d
            # 最后一个学生与最后一个位置的距离
            d = self.N - 1 - self.students[-1]
            if d > dist:
                student = self.N - 1
        # 按顺序插入学生的位置
        bisect.insort(self.students, student)
        return student

    def leave(self, p):
        self.students.remove(p)

"""
382. 链表随机节点
给定一个单链表，随机选择链表的一个节点，并返回相应的节点值。保证每个节点被选的概率一样。
进阶:
如果链表十分大且长度未知，如何解决这个问题？你能否使用常数级空间复杂度实现？
"""

class RandomNode:
    def __init__(self, head: ListNode):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        self.head = head
    def getRandom(self) -> int:
        """
        Returns a random node's value.
        """
        p = self.head
        res = 0
        i = 0
        while(p is not None):
            # 取[0,i]中的随机数
            # 1/i的概率取当前值，(1 - 1/i)的概率保持不变
            if(random.randint(0, i) == 0):
                res = p.val
            i += 1
            p = p.next
        return res

"""
398. 随机数索引
给定一个可能含有重复元素的整数数组，要求随机输出给定的数字的索引。 您可以假设给定的数字一定存在于数组中。
注意：
数组大小可能非常大。 使用太多额外空间的解决方案将不会通过测试。
"""
class RandomIndex:
    def __init__(self, nums: List[int]):
        self.nums = nums
    def pick(self, target: int) -> int:
        n = len(self.nums)
        idxSet = []
        for i in range(n):
            if(target == self.nums[i]):
                idxSet.append(i)
        return random.randint(0, len(idxSet)-1)

class diffArray:
    def __init__(self, nums):
        self.diff = [0 for i in range(len(nums))]
        self.generateDiffArray(nums)
    def generateDiffArray(self, nums):
        if(not nums):
            return
        self.diff[0] = nums[0]
        nLen = len(nums)
        for i in range(1, nLen):
            self.diff[i] = nums[i] - nums[i-1]
    def plusKInSection(self, i, j, k):
        if(i > j):
            return
        self.diff[i] += k
        if((j + 1) < len(self.diff)):
            self.diff[j+1] -= k
    def getResult(self):
        nLen = len(self.diff)
        res = [0 for i in range(nLen)]
        res[0] = self.diff[0]
        for i in range(1, nLen):
            res[i] = res[i-1] + self.diff[i]
        return res

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

"""
341. 扁平化嵌套列表迭代器⭐⭐
给你一个嵌套的整型列表。请你设计一个迭代器，使其能够遍历这个整型列表中的所有整数。
列表中的每一项或者为一个整数，或者是另一个列表。其中列表的元素也可能是整数或是其他列表。
"""
"""
This is the interface that allows for creating nested lists.
You should not implement it, or speculate about its implementation
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

   def getList(self): # -> [NestedInteger]
       """
       @return the nested list that this NestedInteger holds, if it holds a nested list
       Return None if this NestedInteger holds a single integer
       """

class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.retlist = nestedList
    def next(self) -> int:
        first = self.retlist[0].getInteger()
        del(self.retlist[0])
        return first
    def hasNext(self) -> bool:
        while(self.retlist and not self.retlist[0].isInteger()):
            first = self.retlist[0].getList();
            del(self.retlist[0])
            for i in range(len(first)-1, -1, -1):
                item = first[i]
                self.retlist.insert(0, item)
        return len(self.retlist) == 0


"""
380. 常数时间插入、删除和获取随机元素 ⭐⭐
设计一个支持在平均 时间复杂度 O(1) 下，执行以下操作的数据结构。

insert(val)：当元素 val 不存在时，向集合中插入该项。
remove(val)：元素 val 存在时，从集合中移除该项。
getRandom：随机返回现有集合中的一项。每个元素应该有相同的概率被返回。
"""
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # hash映射元素值与索引
        self.val2IdxMap = {}
        # 底层数据结构list
        self.elements = []

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if(val in self.val2IdxMap):
            return False
        self.elements.append(val)
        self.val2IdxMap[val] = len(self.elements) -1
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if(val not in self.val2IdxMap):
            return False
        idx = self.val2IdxMap[val]
        last = len(self.elements) -1
        lastEle = self.elements[last]
        self.elements[last] = val
        self.elements[idx] = lastEle

        self.val2IdxMap[lastEle] = idx

        del self.elements[-1]
        del self.val2IdxMap[val]
        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        max = len(self.elements) - 1
        idx = random.randint(0, max)
        return self.elements[idx]

"""
710. 黑名单中的随机数⭐⭐⭐
给定一个包含 [0，n ) 中独特的整数的黑名单 B，写一个函数从 [ 0，n ) 中返回一个不在 B 中的随机整数。
对它进行优化使其尽量少调用系统方法 Math.random() 。
"""
class BlackList:

    def __init__(self, N: int, blacklist: List[int]):
        self.boundary = N - len(blacklist)
        self.blackMap = {}
        for i in blacklist:
            self.blackMap[i] = -1
        # 将黑名单中的元素映射到区间的尾部
        last = N-1
        for i in self.blacklist:
            # 黑名单成员已经在尾部，不需要映射
            if(i >= self.boundary):
                continue
            # 尾部成员在黑名单中，不能映射
            while (last in self.blackMap):
                last -= 1

            self.blackMap[i] = last
            last -= 1


    def pick(self) -> int:
        idx = random.randint(0, self.boundary-1)
        if(idx in self.blackMap):
            return self.blackMap[idx]
        return idx

class Solution:
    def __init__(self):
        self.ans = float("-inf")
    """
    450. 删除二叉搜索树中的节点 ⭐
    给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。
    返回二叉搜索树（有可能被更新）的根节点的引用。
    """
    def getMinNode(self,root: TreeNode):
        while(root.left is not None):
            root = root.left
        return root

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if(root is None):
            return None
        if(root.val == key):
            # 子节点均为空，直接删除
            if((root.left is None) and (root.right is None)):
                return None
            # 左孩子为None，返回右孩子
            if(root.left is None):
                return root.right
            # 右孩子为空，返回左孩子
            if(root.right is None):
                return root.left
            # 左右孩子均不为空，取左子树的最大节点，或者右子树的最小节点
            minNode = self.getMinNode(root.right)
            root.val = minNode.val
            root.right = self.deleteNode(root.right,minNode.val)
        elif(root.val > key):
            root.left = self.deleteNode(root.left, key)
        elif(root.val < key):
            root.rigth = self.deleteNode(root.right, key)
        return root

    """
    701. 二叉搜索树中的插入操作⭐⭐
    给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 保证原始二叉搜索树中不存在新值。
    注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回任意有效的结果。
    """

    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if(root is None):
            return TreeNode(val)
        if(root.val < val):
            root.right = self.insertIntoBST(root.right, val)
        elif(root.val > val):
            root.left = self.insertIntoBST(root.left, val)
        return root
    """
    700. 二叉搜索树中的搜索⭐
    给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 
    返回以该节点为根的子树。 如果节点不存在，则返回 NULL。
    """
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if(root is None):
            return None
        if(root.val == val):
            return root
        elif(root.val < val):
            return self.searchBST(root.right, val)
        else:
            return self.searchBST(root.left, val)
    """
    226. 翻转二叉树
    翻转一棵二叉树。
    """
    def invertTree(self, root: TreeNode) -> TreeNode:
        if(root is None):
            return root
        tmp = root.left
        root.left = root.right
        root.right = tmp
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
    """
    114. 二叉树展开为链表
    给定一个二叉树，原地将它展开为一个单链表。
    """
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if(root is None):
            return
        self.flatten(root.left)
        self.flatten(root.right)

        right = root.right
        left = root.left

        root.right = left
        root.left = None

        tmp = root
        while(tmp.right is not None):
            tmp = tmp.right
        tmp.right = right
    """
    116. 填充每个节点的下一个右侧节点指针
    给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如
    """
    def connectTwo(self, left: 'None', right: 'None'):
        if(left is None or right is None):
            return None
        left.next = right
        self.connectTwo(left.left, left.right)
        self.connectTwo(right.left, right.right)
        self.connectTwo(left.right, right.left)
    def connect(self, root: 'Node') -> 'Node':
        if(root is None):
            return None
        self.connectTwo(root.left, root.right)
        return root
    """
    316. 去除重复字母⭐⭐
    给你一个仅包含小写字母的字符串，请你去除字符串中重复的字母，使得每个字母只出现一次。
    需保证返回结果的字典序最小（要求不能打乱其他字符的相对位置）。
    """
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        # 保存栈中已存在的字符
        charMap = {}
        # 字符的出现次数
        charCount = [0] * 26
        for c in s:
            charCount[ord(c) - 97] += 1

        for c in s:
            charCount[ord(c) - 97] -= 1
            print(charCount)
            # 已存在
            if (c in charMap and charMap[c]):
                continue
            # 加入，与前一个比较大小
            while (stack and stack[-1] > c):
                # 后面已经没有该字符不删除，否则要删除
                if (charCount[ord(stack[-1]) - 97] == 0):
                    print(stack[-1])
                    break
                # 后面还有可以删除
                charMap[stack.pop()] = False
            stack.append(c)
            charMap[c] = True
        print(charMap)
        ret = "".join(stack)
        return ret
    """
    27. 移除元素
    给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
    不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
    元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
    """
    def removeElement(self, nums: List[int], val: int) -> int:
        slow = 0
        for fast, item in enumerate(nums):
            if(val != nums[fast]):
                nums[slow] = nums[fast]
                slow += 1
        return slow
    """
    283. 移动零
    给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
    """
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        slow = 0
        for fast, item in enumerate(nums):
            if (0 != nums[fast]):
                nums[slow] = nums[fast]
                slow += 1

        for i in range(slow, len(nums)):
            nums[i] = 0
    """
    111. 二叉树的最小深度
    给定一个二叉树，找出其最小深度。
    最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
    说明: 叶子节点是指没有子节点的节点。
    """
    def minDepth(self, root: TreeNode) -> int:
        if(root is None):
            return 0
        depth = 1
        q = Queue()
        q.put(root)
        while(not q.empty()):
            nSize = q.qsize()
            for i in range(nSize):
                ele = q.get()
                if(ele.left is None and ele.right is None):
                    return depth
                if(ele.left is not None):
                    q.put(ele.left)
                if(ele.right is not None):
                    q.put(ele.right)
            depth += 1
        return depth
    """
    752. 打开转盘锁
    你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。
    每个拨轮可以自由旋转：例如把 '9' 变为  '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。
    锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。
    列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。
    字符串 target 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能解锁，返回 -1。
    """
    def openLock(self, deadends: List[str], target: str) -> int:
        deadSet = set(deadends)
        visited = set()
        q = Queue()
        q.put("0000")
        visited.add("0000")
        step = 0
        while(not q.empty()):
            nSize = q.qsize()
            for i in range(nSize):
                ele = q.get()
                if(ele in deadSet):
                    continue
                if(ele == target):
                    return step
                for j in range(4):
                    up = self.minusOne(ele, j)
                    if(up not in visited):
                        q.put(up)
                        visited.add(up)
                    down = self.plusOne(ele, j)
                    if(down not in visited):
                        q.put(down)
                        visited.add(down)
            step += 1
        return -1

    def plusOne(self, pwd, k):
        pwd = list(pwd)
        if(pwd[k] == '9'):
            pwd[k] = '0'
        else:
            pwd[k] = chr(ord(pwd[k])+1)
        return "".join(pwd)
    def minusOne(self, pwd, k):
        pwd = list(pwd)
        if(pwd[k] == '0'):
            pwd[k] = '9'
        else:
            pwd[k] = chr(ord(pwd[k])-1)
        return "".join(pwd)
    """
    704. 二分查找
    给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
    """
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) -1
        while(left <= right):
            # 防止溢出
            mid = int(left + (right-left)/2)
            if(target == nums[mid]):
                return mid
            elif(nums[mid] < target):
                left = mid + 1
            elif(nums[mid] > target):
                right = mid -1
        return -1
    """
    76. 最小覆盖子串
    给你一个字符串 S、一个字符串 T 。请你设计一种算法，可以在 O(n) 的时间复杂度内，从字符串 S 里面找出：包含 T 所有字符的最小子串。
    1、我们在字符串 S 中使用双指针中的左右指针技巧，初始化 left = right = 0，把索引左闭右开区间 [left, right) 称为一个「窗口」。
    2、我们先不断地增加 right 指针扩大窗口 [left, right)，直到窗口中的字符串符合要求（包含了 T 中的所有字符）。
    3、此时，我们停止增加 right，转而不断增加 left 指针缩小窗口 [left, right)，直到窗口中的字符串不再符合要求（不包含 T 中的所有字符了）。同时，每次增加 left，我们都要更新一轮结果。
    4、重复第 2 和第 3 步，直到 right 到达字符串 S 的尽头。
    """
    def minWindow(self, s: str, t: str) -> str:
        need = {}
        window = {}
        for item in t:
            if(item not in need):
                need[item] = 1
            else:
                need[item] += 1
        # 左指针
        left = 0
        # 右指针
        right = 0
        # 子串的起始索引
        start = 0
        # 子串的长度
        nLen = sys.maxsize
        # 记录长度是否符合
        valid = 0
        while(right < len(s)):
            c = s[right]
            right += 1
            if(need.get(c)):
                if(c not in window):
                    window[c] = 1
                else:
                    window[c] += 1
                if(window[c] == need[c]):
                    valid += 1

            while(valid == len(need)):
                if(right - left < nLen):
                    start = left
                    nLen = right - left

                d = s[left]
                left += 1
                if(need.get(d)):
                    if(window[d] == need[d]):
                        valid -= 1
                    window[d] -= 1
        return "" if(nLen == sys.maxsize) else s[start: start + nLen]
    """
    567. 字符串的排列
    给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的排列。
    换句话说，第一个字符串的排列之一是第二个字符串的子串。
    """
    def checkInclusion(self, s1: str, s2: str) -> bool:
        need = {}
        window = {}
        for item in s1:
            if (item not in need):
                need[item] = 1
            else:
                need[item] += 1
        # 左指针
        left = 0
        # 右指针
        right = 0
        # 子串的起始索引
        start = 0
        # 子串的长度
        nLen = sys.maxsize
        # 记录长度是否符合
        valid = 0
        while (right < len(s2)):
            # 移动右指针，加入window
            c = s2[right]
            right += 1
            if (need.get(c)):
                if (c not in window):
                    window[c] = 1
                else:
                    window[c] += 1
                if (window[c] == need[c]):
                    valid += 1
            # 判断window长度是个否超过目标字符串
            while (right - left >= len(s1)):
                if (valid == len(need)):
                    return True
                # 移动左指针，减小window长度
                d = s2[left]
                left += 1
                if (need.get(d)):
                    if (window[d] == need[d]):
                        valid -= 1
                    window[d] -= 1
        return False
    """
    344. 反转字符串
    编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
    不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
    你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。
    输入：["h","e","l","l","o"]
    输出：["o","l","l","e","h"]
    """
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s) - 1
        while(left <= right):
            tmp = s[right]
            s[right] = s[left]
            s[left] = tmp
            left += 1
            right -= 1
    """
    438. 找到字符串中所有字母异位词
    给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。
    字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。
    说明：
    字母异位词指字母相同，但排列不同的字符串。
    不考虑答案输出的顺序。
    """
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 结果列表
        ret = []
        # 目标字符串map
        need = {}
        # 滑动窗口
        window = {}
        for item in p:
            if (item not in need):
                need[item] = 1
            else:
                need[item] += 1
        # 左指针
        left = 0
        # 右指针
        right = 0
        # 子串的起始索引
        start = 0
        # 记录长度是否符合
        valid = 0
        while (right < len(s)):
            c = s[right]
            right += 1
            if (need.get(c)):
                if (c not in window):
                    window[c] = 1
                else:
                    window[c] += 1
                if (window[c] == need[c]):
                    valid += 1

            while (right - left >= len(p)):
                if (valid == len(need)):
                    ret.append(left)

                d = s[left]
                left += 1
                if (need.get(d)):
                    if (window[d] == need[d]):
                        valid -= 1
                    window[d] -= 1
        return ret
    """
    3. 无重复字符的最长子串
    给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
    """
    def lengthOfLongestSubstring(self, s: str) -> int:
        window = {}
        # 左指针
        left = 0
        # 右指针
        right = 0
        # 子串的起始索引
        start = 0
        # 子串的长度
        nLen = sys.maxsize
        ret = 0
        while (right < len(s)):
            c = s[right]
            right += 1

            if (c not in window):
                window[c] = 1
            else:
                window[c] += 1

            while (window[c] > 1):
                d = s[left]
                left += 1
                nLen = max(nLen, right - left)
                window[d] -= 1
        return nLen
    """
    121. 买卖股票的最佳时机
    给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
    如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
    注意：你不能在买入股票前卖出股票。
    """
    def maxProfit(self, prices: List[int]) -> int:
        nLen = len(prices)
        if (nLen == 0):
            return 0
        # 1.状态矩阵
        dp = [[0 for i in range(2)] for j in range(nLen)]

        for i in range(nLen):
            if(i == 0):
                # 第一天，未持有股票，收益为0
                dp[0][0] = 0
                # dp[0][1] = max(dp[-1][1], dp[-1][0] - prices[0])
                dp[0][1] = -prices[0]
                continue
            # 今天未持有股票：昨天未持有，或者昨天持有今天抛出
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            # 今天持有股票：昨天持有，或者昨天未持有但今天买入,只能买入，卖出一次
            dp[i][1] = max(dp[i-1][1], - prices[i])
        return dp[nLen-1][0]
    """
    122. 买卖股票的最佳时机 II
    给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
    设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    """
    def maxProfitII(self, prices: List[int]) -> int:
        nLen = len(prices)
        if(nLen == 0):
            return 0
        # 状态方程
        dp = [[0 for i in range(2)] for j in range(nLen)]

        for i in range(nLen):
            if(i == 0):
                # 第一天，未持有股票，收益为0
                dp[0][0] = 0
                # dp[0][1] = max(dp[-1][1], dp[-1][0] - prices[0])
                dp[0][1] = -prices[0]
                continue
            # 今天未持有股票：昨天未持有，或者昨天持有今天抛出
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            # 今天持有股票：昨天持有，或者昨天未持有但今天买入，可以买卖无数次
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        return dp[nLen-1][0]
    """
    123. 买卖股票的最佳时机 III
    给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
    设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
    注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    """
    def maxProfitIII(self, prices: List[int]) -> int:
        nLen = len(prices)
        if (nLen == 0):
            return 0
        # 状态矩阵nLen*2*2
        dp = [[[0 for i in range(2)] for j in range(3)] for m in range(nLen)]

        for i in range(nLen):
            for k in range(2, 0, -1):
                if (i == 0):
                    # 边界条件
                    # 第一天，未持有股票，收益为0
                    dp[0][k][0] = 0
                    # 第一天，买入，收益为负
                    dp[0][k][1] = -prices[0]
                    continue
                # 今天未持有股票：昨天未持有，或者昨天持有今天抛出
                dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i])
                # 今天持有股票：昨天持有，或者昨天未持有但今天买入，可以买卖无数次
                dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i])
        return dp[nLen - 1][2][0]
    """
    188. 买卖股票的最佳时机 IV
    给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
    设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
    注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    """
    def maxProfitIV(self, k: int, prices: List[int]) -> int:
        nLen = len(prices)
        if (nLen == 0):
            return 0
        if (k > nLen / 2):
            return self.maxProfitInfinity(prices)
        # 状态矩阵nLen*2*2
        dp = [[[0 for i in range(2)] for j in range(k+1)] for m in range(nLen)]

        for i in range(nLen):
            for m in range(0, k + 1):
                if (i == 0):
                    # 边界条件
                    # 第一天，未持有股票，收益为0
                    dp[0][m][0] = 0
                    # 第一天，买入，收益为负
                    dp[0][m][1] = -prices[0]
                    continue
                if (m == 0):
                    dp[i][0][0] = 0
                    dp[i][0][1] = -float("inf")
                    continue
                # 今天未持有股票：昨天未持有，或者昨天持有今天抛出
                dp[i][m][0] = max(dp[i - 1][m][0], dp[i - 1][m][1] + prices[i])
                # 今天持有股票：昨天持有，或者昨天未持有但今天买入，可以买卖无数次
                dp[i][m][1] = max(dp[i - 1][m][1], dp[i - 1][m - 1][0] - prices[i])
        return dp[nLen - 1][k][0]
    """
    309. 最佳买卖股票时机含冷冻期
    给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​
    设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
    1.你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    2.卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
    """
    def maxProfitCold(self, prices: List[int]) -> int:
        nLen = len(prices)
        if (nLen == 0):
            return 0
        # 状态方程，每天有两种状态，持有股票或者未持有股票
        dp = [[0 for i in range(2)] for j in range(nLen)]

        for i in range(nLen):
            if (i == 0):
                # 第一天，未持有股票，收益为0
                dp[0][0] = 0
                # dp[0][1] = max(dp[-1][1], dp[-1][0] - prices[0])
                dp[0][1] = -prices[0]
                continue
            # 今天未持有股票：昨天未持有，或者昨天持有今天抛出
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            # 今天持有股票：昨天持有，或者前天未持有但今天买入(卖出股票后，你无法在第二天买入股票)，可以买卖无数次
            # 注意i<2时，dp[i-2][0]=0, 交易未开始，手中没有股票，故收益为0
            dp[i][1] = max(dp[i - 1][1], dp[i - 2][0] - prices[i])
        return dp[nLen - 1][0]
    """
    714. 买卖股票的最佳时机含手续费
    给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。
    你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
    返回获得利润的最大值。
    注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
    """
    def maxProfitFee(self, prices: List[int], fee: int) -> int:
        nLen = len(prices)
        if (nLen == 0):
            return 0
        # 状态方程
        dp = [[0 for i in range(2)] for j in range(nLen)]

        for i in range(nLen):
            if (i == 0):
                # 第一天，未持有股票，收益为0
                dp[0][0] = 0
                # dp[0][1] = max(dp[-1][1], dp[-1][0] - prices[0])
                dp[0][1] = -prices[0]
                continue
            # 今天未持有股票：昨天未持有，或者昨天持有今天抛出
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            # 今天持有股票：昨天持有，或者昨天未持有但今天买入，可以买卖无数次
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i] - fee)
        return dp[nLen - 1][0]
    """
    ##  102. 二叉树的层序遍历
    给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。
    """
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        q = Queue()
        ret = []
        q.put(root)
        while (not q.empty()):
            size = q.qsize()
            level = []
            for i in range(size):
                item = q.get()
                if (item == None):
                    continue
                level.append(item.val)
                q.put(item.left)
                q.put(item.right)
            if (level):
                ret.append(level)
        return ret
    """
    198. 打家劫舍
    你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
    如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
    给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
    """
    def rob(self, nums: List[int]) -> int:
        nLen = len(nums)
        if(nLen == 0):
            return 0
        # 状态矩阵
        dp = [[0, 0] for i in range(nLen)]

        for i in range(nLen):
            if(i == 0):
                dp[0][0] = 0
                dp[0][1] = nums[0]
                continue
            dp[i][0] = max(dp[i-1][0], dp[i-1][1])
            dp[i][1] = dp[i-1][0] + nums[i]
        return max(dp[nLen-1][0], dp[nLen-1][1])
    """
    213. 打家劫舍 II
    你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都围成一圈，
    这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，
    系统会自动报警。
    给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。
    """
    def robII(self, nums: List[int]) -> int:
        nLen = len(nums)
        if (nLen == 1):
            return nums[0]
        nums1 = nums[0: nLen-1]
        nums2 = nums[1: nLen]
        return max(self.rob(nums1), self.rob(nums2))
    """
    337. 打家劫舍 III
    在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 
    除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 
    如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
    计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
    """
    memMap = {}
    def robIII(self, root: TreeNode) -> int:
        if(root is None):
            return 0
        if(root in Solution.memMap):
            return Solution.memMap[root]
        # 抢
        rob_it = root.val \
                 + (0 if(root.left is None) else self.robIII(root.left.left) + self.robIII(root.left.right)) \
                 + (0 if(root.right is None) else self.robIII(root.right.left) + self.robIII(root.right.right))
        # 不抢
        not_rob = self.robIII(root.left) + self.robIII(root.right)

        ret = max(rob_it, not_rob)

        Solution.memMap[root] = ret

        return ret
    """
    1288. 删除被覆盖区间
    给你一个区间列表，请你删除列表中被其他区间所覆盖的区间。
    只有当 c <= a 且 b <= d 时，我们才认为区间 [a,b) 被区间 [c,d) 覆盖。
    在完成所有删除操作后，请你返回列表中剩余区间的数目。
    """
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        # 排序，按起点升序排列，起点相同，终点降序排列
        def compareRule(a, b):
            if(a[0] == b[0]):
                return b[1] - a[1]
            return a[0] - b[0]
        intervals.sort(key=functools.cmp_to_key(compareRule))

        # 记录区间的起点和终点
        start = intervals[0][0]
        end = intervals[0][1]
        ret = 0
        for i in range(1, len(intervals)):
            item = intervals[i]
            s = item[0]
            e = item[1]
            # 包含关系
            if(start <=s and end >= e):
                ret += 1
            # 相交
            if(end >= s and end <= e):
                end = e
            # 不相交
            if(end <= s):
                start = s
                end = e
        return ret
    """
    56. 合并区间
    给出一个区间的集合，请合并所有重叠的区间。
    """
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # :cry:
        if(intervals is None or len(intervals) == 0):
            return []
        # 区间按起点升序排列，
        # 起点相同，按终点降序排列
        def compareRelu(a,b):
            if(a[0] == b[0]):
                return b[1] - a[1]
            return a[0] - b[0]
        intervals.sort(key=functools.cmp_to_key(compareRelu))

        ret = []
        ret.append(intervals[0])
        for i in range(1, len(intervals)):
            cur = intervals[i]
            last = ret[-1]
            # 区间相交
            if(cur[0] <= last[1]):
                last[1] = max(cur[1], last[1])
            else:
                ret.append(cur)
        return ret
    """
    986. 区间列表的交集
    给定两个由一些 闭区间 组成的列表，每个区间列表都是成对不相交的，并且已经排序。
    返回这两个区间列表的交集。
    （形式上，闭区间 [a, b]（其中 a <= b）表示实数 x 的集合，而 a <= x <= b。两个闭区间的交集是一组实数，
    要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3]。）
    """
    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        # 遍历两个区间组合
        if(A is None):
            return B
        if(B is None):
            return A
        ret = []
        # 双指针
        i = 0
        j = 0
        while(i < len(A) and j < len(B)):
            aStart, aEnd = A[i][0], A[i][1]
            bStart, bEnd = B[j][0], B[j][1]
            # 有交集
            if(bEnd >= aStart and aEnd >= bStart):
                ret.append([max(aStart, bStart), min(aEnd, bEnd)])
            # 不相交，指针前进
            if(bEnd < aEnd):
                j += 1
            else:
                i += 1
        return ret
    """
    15. 三数之和
    给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？
    请你找出所有满足条件且不重复的三元组。
    注意：答案中不可以包含重复的三元组。
    """
    def twoSum(self, nums: List[int], target: int):
        ret = []
        nLen = len(nums)
        # 双指针
        left = 0
        right = nLen - 1
        while(left < right):
            nL, nR = nums[left], nums[right]
            nSum = nL + nR
            if(nSum < target):
                while(left < right and nums[left] == nL):
                    left += 1
            elif(nSum > target):
                while(left < right and nums[right] == nR):
                    right -= 1
            elif(nSum == target):
                ret.append([nL, nR])
                while(left < right and nL == nums[left]):
                    left += 1
                while(left < right and nR == nums[right]):
                    right -= 1
        return ret
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nLen = len(nums)
        ret = []
        if(nLen == 0):
            return ret
        nums.sort()
        i = 0
        while(i < nLen):
            f = nums[i]
            twoRet = self.twoSum(nums[i+1:], 0 - f)
            for idx, item in enumerate(twoRet):
                ret.append([f, item[0], item[1]])
            while(i < nLen and nums[i] == f):
                i += 1
        return ret
    """
    887. 鸡蛋掉落
    你将获得 K 个鸡蛋，并可以使用一栋从 1 到 N  共有 N 层楼的建筑。
    每个蛋的功能都是一样的，如果一个蛋碎了，你就不能再把它掉下去。
    你知道存在楼层 F ，满足 0 <= F <= N 任何从高于 F 的楼层落下的鸡蛋都会碎，从 F 楼层或比它低的楼层落下的鸡蛋都不会破。
    每次移动，你可以取一个鸡蛋（如果你有完整的鸡蛋）并把它从任一楼层 X 扔下（满足 1 <= X <= N）。
    你的目标是确切地知道 F 的值是多少。
    无论 F 的初始值如何，你确定 F 的值的最小移动次数是多少？
    """
    def superEggDrop(self, K: int, N: int) -> int:
        memo = {}
        def dp(K: int, N: int) -> int:
            # 边界条件
            # 只有一个鸡蛋，只能一层一层的试
            if(K == 1):
                return N
            if(N == 0):
                return 0
            if((K, N) in memo):
                return memo[(K, N)]
            ret = float("inf")
            # for i in range(1, N + 1):
            #     # 所有坏的情况中找一个最好的
            #     ret = min(ret, max(
            #         # 鸡蛋没碎
            #         dp(K, N - i),
            #         # 鸡蛋碎了
            #         dp(K - 1, i - 1)
            #     ) + 1) # 第i层扔了一次
            lo , hi = 1, N
            while(lo <= hi):
                mid = lo + (hi-lo)//2
                # 单调递增 f(K, mid) = dp(K, mid-1)
                broken = dp(K-1, mid-1)
                # 单调递减 f(K, mid) = dp(K, N-mid)
                unbroken = dp(K, N-mid)
                if(broken > unbroken):
                    hi = mid - 1
                    ret = min(ret, broken+1)
                else:
                    lo = mid + 1
                    ret = min(ret, unbroken + 1)
            # 备忘录
            memo[(K, N)] = ret
            return ret
        return dp(K, N)
    """
    654. 最大二叉树
    给定一个不含重复元素的整数数组。一个以此数组构建的最大二叉树定义如下：
    二叉树的根是数组中的最大元素。
    左子树是通过数组中最大值左边部分构造出的最大二叉树。
    右子树是通过数组中最大值右边部分构造出的最大二叉树。
    通过给定的数组构建最大二叉树，并且输出这个树的根节点。
    """
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        if(nums is None or len(nums) == 0):
            return None
        maxN = max(nums)
        maxIdx = nums.index(maxN)
        node = TreeNode(maxN)
        node.left = self.constructMaximumBinaryTree(nums[:maxIdx])
        node.right = self.constructMaximumBinaryTree(nums[maxIdx+1:])
        return node
    """
    105. 从前序与中序遍历序列构造(还原)二叉树
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
    def buildHelper(self, preorder: List[int], prestart: int, preend: int, inorder: List[int], instart: int, inend: int):
        if(prestart > preend or instart > inend):
            return None
        # 根节点
        rootVal = preorder[prestart]
        # 根节点的索引
        rootIdx = inorder.index(rootVal)

        root = TreeNode(rootVal)
        leftSize = rootIdx - instart
        root.left = self.buildHelper(preorder, prestart+1, prestart + leftSize, inorder, instart, rootIdx-1)
        root.right = self.buildHelper(preorder, prestart + leftSize + 1, preend, inorder, rootIdx + 1, inend)
        return root
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        end = len(preorder)
        root = self.buildHelper(preorder, 0, end-1, inorder, 0, end-1)
        return root
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
    def buildHelperII(self, inorder: List[int], instart: int, inend: int, postorder: List[int], poststart: int, postend: int):
        if(instart > inend or poststart > postend):
            return None
        rootVal = postorder[postend]
        rootIdx = inorder.index(rootVal)

        leftSize = rootIdx - instart

        root = TreeNode(rootVal)
        root.left = self.buildHelperII(inorder, instart, rootIdx-1, postorder, poststart, poststart+leftSize-1)
        root.right = self.buildHelperII(inorder, rootIdx+1, inend, postorder, poststart+leftSize, postend-1)
        return root
    def buildTreeII(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        end = len(inorder)
        root = self.buildHelperII(inorder, 0, end-1, postorder, 0, end-1)
        return root
    """
    652. 寻找重复的子树
    给定一棵二叉树，返回所有重复的子树。对于同一类的重复子树，你只需要返回其中任意一棵的根结点即可。
    两棵树重复是指它们具有相同的结构以及相同的结点值。
    """
    def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
        # 保存重复的子树字符串
        memo = dict()
        res = []
        def postTravse(root: TreeNode):
            if(root is None):
                return "#"
            # 后序遍历
            left = postTravse(root.left)
            right = postTravse(root.right)
            # 子树 后序遍历为 字符串
            subTreeStr = left + "," + right + "," + str(root.val)

            freq = memo.get(subTreeStr, 0)
            if(freq == 1):
                res.append(root)
            memo[subTreeStr] = freq + 1
            return subTreeStr
        postTravse(root)
        return res
    """
    416. 分割等和子集
    给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
    注意:
    每个数组中的元素不会超过 100
    数组的大小不会超过 200
    """
    def canPartition(self, nums: List[int]) -> bool:
        # 转化为背包问题
        nSum = sum(nums)
        nLen = len(nums)
        # 不是偶数说明不能划分
        if(nSum%2 != 0):
            return False
        # 能够划分则说明，nums中的元素能够凑成 nSum/2
        nSum = nSum/2
        # 边界条件
        dp = [[False for i in range(nSum + 1)] for j in range(nLen + 1)]
        for i in range(nLen + 1):
            dp[i][0] = True

        for i in range(1, nLen + 1):
            for j in range(1, nSum + 1):
                if (j - nums[i - 1] < 0):
                    # 无法加入
                    dp[i][j] = dp[i - 1][j]
                else:
                    # 可以加入
                    dp[i][j] = dp[i-1][j-nums[i-1]] or dp[i-1][j]
        return dp[nLen][nSum]
    """
    518. 零钱兑换 II
    给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。 
    """
    def change(self, amount: int, coins: List[int]) -> int:
        # 类似于背包问题， 状态有两种，对于前i个硬币，凑出j总金额，有dp[i][j]种凑法
        nLen = len(coins)
        # 边界条件，dp[...][0]=1 dp[0][...]=0
        dp = [[0 for i in range(amount+1)] for j in range(nLen+1)]

        for i in range(0, nLen+1):
            dp[i][0] = 1

        for i in range(1, nLen+1):
            for j in range(1, amount+1):
                if(j-coins[i-1] >= 0):
                    # 使用前i个面额的硬币凑出总金额j的组合数
                    # 使用当前面额和不使用当前面额凑出j的总数
                    dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[nLen][amount]
    """
    55. 跳跃游戏
    给定一个非负整数数组，你最初位于数组的第一个位置。
    数组中的每个元素代表你在该位置可以跳跃的最大长度。
    判断你是否能够到达最后一个位置。
    """
    def canJump(self, nums: List[int]) -> bool:
        nLen = len(nums)
        jumpDistance = 0
        # 计算跳到倒数第二个位置时，能跳的最大距离
        for i in range(nLen-1):
            # 计算每一步所能跳的最大距离
            jumpDistance = max(jumpDistance, i + nums[i])
            if(jumpDistance <= i):
                return False
        # 跳跃的最大距离能否超过数组的最后一个位置
        return jumpDistance >= nLen-1
    """
    45. 跳跃游戏 II
    给定一个非负整数数组，你最初位于数组的第一个位置。
    数组中的每个元素代表你在该位置可以跳跃的最大长度。
    你的目标是使用最少的跳跃次数到达数组的最后一个位置。
    """
    # 动态规划,会超时
    def jumpI(self, nums: List[int]) -> int:
        nLen = len(nums)
        dp = [nLen + 1 for i in range(nLen)]
        # 已经在最后一个位置，需要跳跃0次
        dp[nLen - 1] = 0

        for i in range(nLen - 2, -1, -1):
            steps = nums[i]
            need = nLen - 1 - i
            for j in range(1, steps + 1):
                if (need <= j):
                    dp[i] = 1
                    break
                else:
                    # 跳到后面的某一个（i+j）位置，得到到达最后一个位置需要的最少次数
                    dp[i] = min(dp[i + j] + 1, dp[i])
        return dp[0]
    # 贪心策略
    def jumpII(self, nums: List[int]) -> int:
        nLen = len(nums)
        # 能跳到的最远的位置
        end = 0
        jumpDistance = 0
        j = 0

        for i in range(nLen - 1):
            # 选择从初始位置到当前位置中，能跳的最远距离
            jumpDistance = max(jumpDistance, nums[i] + i)
            # 到达之后需要再跳一次，继续跳的尽量远
            if (i == end):
                j += 1
                end = jumpDistance
        return j
    """
    877. 石子游戏
    亚历克斯和李用几堆石子在做游戏。偶数堆石子排成一行，每堆都有正整数颗石子 piles[i] 。
    游戏以谁手中的石子最多来决出胜负。石子的总数是奇数，所以没有平局。
    亚历克斯和李轮流进行，亚历克斯先开始。 每回合，玩家从行的开始或结束处取走整堆石头。 这种情况一直持续到没有更多的石子堆为止，此时手中石子最多的玩家获胜。
    假设亚历克斯和李都发挥出最佳水平，当亚历克斯赢得比赛时返回 true ，当李赢得比赛时返回 false 。
    """
    def stoneGame(self, piles: List[int]) -> bool:
        nLen = len(piles)
        # 元组第一个元素表示先手，第二个元素表示后手
        dp = [[[0, 0] for i in range(nLen)] for i in range(nLen)]

        # 边界条件
        for i in range(nLen):
            dp[i][i][0] = piles[i]
            dp[i][i][1] = 0

        # 沿对角线遍历数组
        for l in range(2, nLen + 1):
            for i in range(nLen - l + 1):
                j = l + i - 1
                # 取最左边的一堆,所能得到的石头总数
                left = piles[i] + dp[i + 1][j][1]
                # 取最右边的一堆,所能得到的石头总数
                right = piles[j] + dp[i][j - 1][1]

                if (left > right):
                    dp[i][j][0] = left
                    dp[i][j][1] = dp[i + 1][j][0]
                else:
                    dp[i][j][0] = right
                    dp[i][j][1] = dp[i][j - 1][0]
        return dp[0][nLen - 1][0] > dp[0][nLen - 1][1]
    """
    10. 正则表达式匹配
    给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
    '.' 匹配任意单个字符
    '*' 匹配零个或多个前面的那一个元素
    所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。
    """
    mem = dict()
    def dypro(self, s, i, p, j):
        nS = len(s)
        nP = len(p)
        # 边界条件
        if(j == nP):
            return i == nS
        if(i == nS):
            if((nP-j)%2 == 1):
                return False
            else:
                for k in range(j, nP-1, 2):
                    if(p[k+1] != '*'):
                        return False
                return True
        key = str(i) + ',' + str(j)
        if(key in Solution.mem):
            return Solution.mem[key]

        res = False
        # 匹配的情况
        if(s[i] == p[j] or p[j] == '.'):
            if(j < nP-1 and p[j+1] == '*'):
                # 匹配零次或者多次
                res = self.dypro(s, i, p, j+2) or self.dypro(s, i+1, p, j)
            else:
                res = self.dypro(s, i+1, p, j+1)
        # 不匹配
        else:
            if(j < nP-1 and p[j+1] == '*'):
                # 此时，只能匹配零次
                res = self.dypro(s, i, p, j+2)
            else:
                res = False
        Solution.mem[key] = res
        return res
    def isMatch(self, s: str, p: str) -> bool:
        return self.dypro(s, 0, p, 0)
    """
    28. 实现 strStr()
    给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。
    如果不存在，则返回  -1
    """
    def strStr_bruteforce(self, haystack: str, needle: str) -> int:
        N = len(haystack)
        M = len(needle)
        if (M == 0):
            return 0
        for i in range(N - M + 1):
            j = 0
            matched = True
            for j in range(M):

                if (haystack[i + j] != needle[j]):
                    matched = False
                    break
            if (matched and j + 1 == M):
                return i
        return -1
    """
    1312. 让字符串成为回文串的最少插入次数
    给你一个字符串 s ，每一次操作你都可以在字符串的任意位置插入任意字符。
    请你返回让 s 成为回文串的 最少操作次数 。
    「回文串」是正读和反读都相同的字符串。
    """
    def minInsertions(self, s: str) -> int:
        nLen = len(s)
        if(nLen == 0):
            return 0
        dp = [[0 for i in range(nLen)] for i in range(nLen)]
        # 边界条件，i==j时，dp[i][j] = 0
        for i in range(nLen-2, -1, -1):
            for j in range(i+1, nLen):
                if(s[i] == s[j]):
                    dp[i][j] = dp[i+1][j-1]
                else:
                    dp[i][j] = min(dp[i+1][j], dp[i][j-1]) + 1
        return dp[0][nLen-1]
    """
    234. 回文链表
    请判断一个链表是否为回文链表。
    """
    def reverseSingleList(self, head: ListNode) -> ListNode:
        pre = None
        cur = head
        while(cur is not None):
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
        return pre
    def isPalindrome(self, head: ListNode) -> bool:
        # 找到链表的中点，双指针
        slow = head
        fast = head
        while(fast is not None and fast.next is not None):
            slow = slow.next
            fast = fast.next.next
        # 偶数个节点
        if(fast is not None):
            slow = slow.next
        # 将后半部分反转
        right = self.reverseSingleList(slow)
        left = head
        while(right is not None):
            if(right.val != left.val):
                return False
            right = right.next
            left = left.next
        return True
    """
    516. 最长回文子序列
    给定一个字符串 s ，找到其中最长的回文子序列，并返回该序列的长度。可以假设 s 的最大长度为 1000
    """
    def longestPalindromeSubseq(self, s: str) -> int:
        nLen = len(s)
        if(nLen == 0):
            return 0
        # dp[i][j]表示s[i] 到 s[j]最长的回文字串的长度
        dp = [[0 for i in range(nLen)] for i in range(nLen)]
        # 边界条件 i == j 时，回文子串的长度为1
        for i in range(nLen):
            dp[i][i] = 1
        # dp table 从下到上，从左到右遍历
        # j > i
        for i in range(nLen-2, -1, -1):
            for j in range(i+1, nLen):
                if(s[i] == s[j]):
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return dp[0][nLen-1]
    """
    230. 二叉搜索树中第K小的元素
    给定一个二叉搜索树，编写一个函数 kthSmallest 来查找其中第 k 个最小的元素。
    说明：
    你可以假设 k 总是有效的，1 ≤ k ≤ 二叉搜索树元素个数。
    """
    def postTravel(self, root: TreeNode, ret: [int]):
        if(root is None):
            return ret
        self.postTravel(root.left, ret)
        ret.append(root.val)
        self.postTravel(root.right, ret)
        return ret
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        ret = []
        self.postTravel(root, ret)
        return ret[k-1]
    """
    538. 把二叉搜索树转换为累加树
    给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。
    提醒一下，二叉搜索树满足下列约束条件：
    节点的左子树仅包含键 小于 节点键的节点。
    节点的右子树仅包含键 大于 节点键的节点。
    左右子树也必须是二叉搜索树。
    """
    sum = 0
    # 降序遍历
    def descendTravel(self, root: TreeNode):
        if (root is None):
            return
        self.descendTravel(root.right)
        Solution.sum += root.val
        root.val = Solution.sum
        self.descendTravel(root.left)

    def convertBST(self, root: TreeNode) -> TreeNode:
        self.descendTravel(root)
        return root
    """
    560. 和为K的子数组
    给定一个整数数组和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。
    """
    def subarraySum(self, nums: List[int], k: int) -> int:
        nLen = len(nums)
        # 前缀和{前缀和：出现次数}
        prefixSum = dict()
        prefixSum.update({0: 1})

        sum_i = 0
        sum_j = 0
        ans = 0
        for i in range(nLen):
            # 前i+1个元素的和
            sum_i += nums[i]
            # sum_i - sum_j = k
            sum_j = sum_i - k
            if(sum_j in prefixSum):
                ans += prefixSum.get(sum_j)
            # 记录出现的前缀和
            prefixSum.update({sum_i: prefixSum.get(sum_i, 0) + 1})
        return ans
    """
    1109. 航班预订统计
    这里有 n 个航班，它们分别从 1 到 n 进行编号。
    我们这儿有一份航班预订表，表中第 i 条预订记录 bookings[i] = [j, k, l] 意味着我们在从 j 到 k 的每个航班上预订了 l 个座位。
    请你返回一个长度为 n 的数组 answer，按航班编号顺序返回每个航班上预订的座位数。
    """
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        nLen = len(bookings)
        res = [0 for i in range(n)]
        diff = diffArray(res)
        for m in range(nLen):
            item = bookings[m]
            i = item[0]-1
            j = item[1]-1
            k = item[2]
            diff.plusKInSection(i, j, k)
        res = diff.getResult()
        return res
    """
    461. 汉明距离
    两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
    给出两个整数 x 和 y，计算它们之间的汉明距离。
    """
    def hammingWeight(self, x: int) -> int:
        res = 0
        while (x != 0):
            # 消除x的二进制中的最后一个1
            x = x & (x - 1)
            # 自加1
            res = -~res
        return res

    def hammingDistance(self, x: int, y: int) -> int:
        tmp = x ^ y
        res = self.hammingWeight(tmp)
        return res
    """
    136. 只出现一次的数字
    给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
    说明：
    你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
    """
    def singleNumber(self, nums: List[int]) -> int:
        # 一个数与自己异或是0，与0异或保持不变
        res = 0
        nLen = len(nums)
        for i, n in enumerate(nums):
            res ^= n
        return res
    """
    231. 2的幂
    给定一个整数，编写一个函数来判断它是否是 2 的幂次方。
    """
    def isPowerOfTwo(self, n: int) -> bool:
        if (n <= 0):
            return False
        # 2的幂二进制有且仅有一个1
        return (n & (n - 1)) == 0
    """
    43. 字符串相乘
    给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
    """
    def multiply(self, num1: str, num2: str) -> str:
        nLen1 = len(num1)
        nLen2 = len(num2)
        res = [0 for i in range(nLen1 + nLen2)]
        # 逆序遍历
        for i in range(nLen1-1, -1, -1):
            for j in range(nLen2-1, -1, -1):
                mul = int(num1[i]) * int(num2[j])
                p1 = i+j
                p2 = i+j+1
                sum = mul + res[p2]
                o = sum%10
                t = sum//10
                res[p2] = o
                res[p1] += t
        # 去掉前缀0
        i = 0
        while(i < len(res) and res[i] == 0):
            i += 1
        res_ = []
        for j in range(i, len(res)):
            res_.append(str(res[j]))
        if(len(res_) == 0):
            return '0'
        return ''.join(res_)

    """
    172. 阶乘后的零
    给定一个整数 n，返回 n! 结果尾数中零的数量。
    """
    def trailingZeroes(self, n: int) -> int:
        # 取决于n可以分解为2*5的个数
        # 其中2的个数要多于5的个数
        # 进一步，n可以分解为多少个因子5，25，125......
        divisor = 5
        res = 0
        while(divisor <= n):
            res += n//divisor
            divisor *= 5
        return res
    """
    793. 阶乘函数后K个零
    f(x) 是 x! 末尾是0的数量。（回想一下 x! = 1 * 2 * 3 * ... * x，且0! = 1）
    例如， f(3) = 0 ，因为3! = 6的末尾没有0；而 f(11) = 2 ，因为11!= 39916800末端有2个0。给定 K，找出多少个非负整数x ，有 f(x) = K 的性质。
    """
    def lowerBound(self, K: int):
        lo = 10**K
        up = 10**(K+1)
        mid = 0
        while(lo < up):
            mid = lo + (up-lo)//2
            if(self.trailingZeroes(mid) < K):
                lo = mid +1
            elif(self.trailingZeroes(mid) > K):
                up = mid
            else:
                up = mid
        return lo
    def upperBound(self, K: int):
        lo = 10**K
        up = 10**(K+1)
        mid = 0
        while(lo < up):
            mid = lo + (up-lo)//2
            if(self.trailingZeroes(mid) < K):
                lo = mid +1
            elif(self.trailingZeroes(mid) > K):
                up = mid
            else:
                lo = mid + 1
        return up
    def preimageSizeFZF(self, K: int) -> int:
        return self.upperBound(K) - self.lowerBound(K) +1
    """
    448. 找到所有数组中消失的数字
    给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
    找到所有在 [1, n] 范围之间没有出现在数组中的数字。
    您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。
    """
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        s = set(nums)
        n = len(nums)
        res = []
        for i in range(n):
            if(i not in s):
                res.append(i)

        return res
    """
    645. 错误的集合
    集合 S 包含从1到 n 的整数。不幸的是，因为数据错误，导致集合里面某一个元素复制了成了集合里面的另外一个元素的值，导致集合丢失了一个整数并且有一个元素重复。
    给定一个数组 nums 代表了集合 S 发生错误后的结果。你的任务是首先寻找到重复出现的整数，再找到丢失的整数，将它们以数组的形式返回。
    """
    def findErrorNums(self, nums: List[int]) -> List[int]:
        n = len(nums)
        dum = -1
        # 将索引所对应的元素取负，如果已经为负，说明这个元素是重复元素
        for i in range(n):
            idx = abs(nums[i]) - 1
            if (nums[idx] < 0):
                dum = abs(nums[i])
            else:
                nums[idx] *= -1
        miss = -2
        for i in range(n):
            if (nums[i] > 0):
                miss = i + 1
        return [dum, miss]
    """
    224. 基本计算器
    实现一个基本的计算器来计算一个简单的字符串表达式的值。
    字符串表达式可以包含左括号 ( ，右括号 )，加号 + ，减号 -，非负整数和空格  。
    """
    def calculate(self, s: str) -> int:
        """
        +1+2-3
        前面添加+，使符号与数字成对
        有（）表示一个计算单元，利用递归来计算
        :param s:
        :return:
        """
        def helper(substr: List) -> int:
            if(substr is None):
                return 0
            num = 0
            sign = '+'
            stack = []
            while(substr):
                s0 = substr.pop(0)
                isD = s0.isdigit()
                if(isD):
                    num = num*10 + int(s0)
                # 右括号开始递归
                if(s0 == '('):
                    num = helper(substr)
                # 不是数字并且不是空格 或者 长度为0
                if((not isD and s0 != ' ') or len(substr) == 0):
                    if(sign == '+'):
                        stack.append(num)
                    elif(sign == '-'):
                        stack.append(-num)
                    elif(sign == '*'):
                        last = stack.pop()
                        n = last * num
                        stack.append(n)
                    elif(sign == '/'):
                        last = stack.pop()
                        n = last//num
                        stack.append(n)
                    sign = s0
                    num = 0
                # 左括号递归结束
                if (s0 == ')'):
                    break
            return sum(stack)
        return helper(list(s))
    """
    42. 接雨水
    给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
    """
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if(n == 0):
            return 0
        l_max = height[0]
        r_max = height[n-1]
        left = 0
        right = n-1
        res = 0
        while(left <= right):
            l_max = max(l_max, height[left])
            r_max = max(r_max, height[right])

            if(l_max < r_max):
                res += (l_max - height[left])
                left += 1
            else:
                res += (r_max - height[right])
                right -= 1
        return res

    """
    20. 有效的括号
    给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
    """
    def leftBrackets(self, c):
        right = ''
        if(c == ')'):
            right = '('
        elif(c == ']'):
            right = '['
        elif(c == '}'):
            right = '{'
        return right
    def isValid(self, s: str) -> bool:
        stack = []

        for idx, c in enumerate(s):
            if(c == '(' or c == '{' or c == '['):
                stack.append(c)
            else:
                if(stack and self.leftBrackets(c) == stack[-1]):
                    stack.pop()
                else:
                    return False
        return len(stack) == 0

    """
    392. 判断子序列
    给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
    字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
    """
    def isSubsequence(self, s: str, t: str) -> bool:
        ns = len(s)
        nt = len(t)
        i = 0
        j = 0
        while(i < ns and j < nt):
            if(s[i] == t[j]):
                i += 1
            j += 1
        return i == ns

    def leftBound(self, arr, target):
        lo = 0
        hi = len(arr)

        while(lo < hi):
            mid = lo + (hi-lo)//2
            if(target > arr[mid]):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def isSubsequenceByDivision(self, s: str, t: str) -> bool:
        index = dict()
        # 字典记录字符出现的索引
        for i, c in enumerate(t):
            if(c not in index):
                index[c] = [i]
            else:
                index[c].append(i)

        i = 0
        for j, c in enumerate(s):
            if(c not in index):
                return False
            else:
                idxs = index[c]
                # 二分法查找字符c大于i的索引
                lo = self.leftBound(idxs, i)
                if(lo == len(idxs)):
                    return False
                else:
                    # 在t上移动索引i
                    i = idxs[lo] + 1
        return True
    """
    124. 二叉树中的最大路径和
    给定一个非空二叉树，返回其最大路径和。
    本题中，路径被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。
    """
    def postTravelSum(self, root: TreeNode) -> int:
        if(root is None):
            return 0
        # 左子树的最大路径和，为负则对整体没有贡献，不必加到路径和中
        left = max(0, self.postTravelSum(root.left))
        right = max(0, self.postTravelSum(root.right))
        # 如果均为负，则最大路径和为父节点的和
        self.ans = max(self.ans, left + right + root.val)
        return max(left, right) + root.val
    def maxPathSum(self, root: TreeNode) -> int:
        self.postTravelSum(root)
        return self.ans
    """
    99. 恢复二叉搜索树
    给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。
    进阶：使用 O(n) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？
    """
    def recoverTree(self, root: TreeNode) -> None:
        """
        中序遍历，节点值依次递增，前一个节点的值小与后一个节点的值，否则，说明位置错误
        """
        self.prev = TreeNode(float("-inf"))
        self.error1 = None
        self.error2 = None
        def inOrder(root: TreeNode) -> None:
            if (root is None):
                return
            inOrder(root.left)
            if(self.error1 is None and self.prev.val > root.val):
                self.error1 = self.prev
            if(self.error1 and self.prev.val > root.val):
                self.error2 = root
            self.prev = root
            inOrder(root.right)
        inOrder(root)
        self.error1.val, self.error2.val = self.error2.val, self.error1.val
    """
    354. 俄罗斯套娃信封问题
    给定一些标记了宽度和高度的信封，宽度和高度以整数对形式 (w, h) 出现。当另一个信封的宽度和高度都比这个信封大的时候，
    这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
    请计算最多能有多少个信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。
    说明:
    不允许旋转信封。
    """
    def lengthOfLISDichotomy(self, nums: List[int]):
        piles = 0
        n = len(nums)
        top = [0 for _ in range(n)]
        for i in range(n):
            poker = nums[i]
            left = 0
            right = piles
            # 二分查找插入位置
            while (left < right):
                mid = (left + right) / 2;
                if (top[mid] >= poker):
                    right = mid;
                else:
                    left = mid + 1;
            if (left == piles):
                piles += 1
            # 把这张牌放到牌堆顶
            top[left] = poker;
        return piles;

    def lengthOfLISDP(self, nums: List[int]):
        n = len(nums)
        if(n == 0):
            return 0
        # 边界条件
        dp = [1 for _ in range(n)]
        for i in range(n):
            for j in range(i):
                if(nums[i] > nums[j]):
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)

    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # 排序，按w升序排列，w相同，h降序排列
        def compareRule(a, b):
            if (a[0] == b[0]):
                return b[1] - a[1]
            return a[0] - b[0]
        envelopes.sort(key=functools.cmp_to_key(compareRule))

        h = []
        for idx,i in enumerate(envelopes):
            h.append(i[1])

        return self.lengthOfLISDP(h)
    """
    53. 最大子序和
    给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
    """
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        if(n==0):
            return 0
        dp = [[(-sys.maxsize - 1) for _ in range(n)] for _ in range(n)]
        # 边界条件
        for i in range(n):
            dp[i][i] = nums[i]

        for i in range(n):
            for j in range(i+1, n):
                dp[i][j] = dp[i][j-1] + nums[j]

        return max(dp)
    def maxSubArrayII(self, nums: List[int]) -> int:
        n = len(nums)
        if (n == 0):
            return 0
        dp = [(-sys.maxsize - 1) for _ in range(n)]
        # 边界条件
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(nums[i], nums[i] + dp[i-1])
        return max(dp)
    """
    25. K 个一组翻转链表
    给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
    k 是一个正整数，它的值小于或等于链表的长度。
    如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
    """
    # 反转区间[start, end)之间的链表
    def reverseSegment(self, start: ListNode, end: ListNode):
        cur = start
        nxt = start
        pre = None
        while(cur != end):
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        # 返回新的头节点
        return pre

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if(head is None):
            return head
        a = head
        b = head
        # 递归条件，不足k个直接返回
        for i in range(k):
            if(b is None):
                return head
            b = b.next

        new = self.reverseSegment(a, b)

        a.next = self.reverseKGroup(b, k)

        return new
    """
    130. 被围绕的区域
    给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。
    找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
    """
    def solve(self, board: List[List[str]]) -> None:
        """
        主要思路是适时增加虚拟节点，想办法让元素「分门别类」，建立动态连通关系。
        """
        xLen = len(board)
        if(xLen == 0):
            return
        yLen = len(board[0])

        uf = UF(xLen * yLen + 1)
        dummy = xLen * yLen

        for i in range(xLen):
            if(board[i][0] == 'O'):
                uf.union(i * yLen, dummy)
            if(board[i][yLen-1] == 'O'):
                uf.union(i * yLen + yLen -1, dummy)
        for i in range(yLen):
            if(board[0][i] == 'O'):
                uf.union(i, dummy)
            if(board[xLen-1][i] == 'O'):
                uf.union((xLen-1) * yLen + i, dummy)

        d = [[1,0], [0,1], [0,-1], [-1,0]]

        for i in range(1, xLen-1):
            for j in range(1, yLen-1):
                if(board[i][j] == 'O'):
                    for k in range(4):
                        x = i + d[k][0]
                        y = j + d[k][1]
                        if(board[x][y] == 'O'):
                            uf.union(x * yLen + y, i * yLen +j)

        for i in range(1, xLen-1):
            for j in range(1, yLen-1):
                if(not uf.connected(i * yLen + j, dummy)):
                    board[i][j] = 'X'
    """
    990. 等式方程的可满足性
    给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations[i] 的长度为 4，
    并采用两种不同的形式之一："a==b" 或 "a!=b"。在这里，a 和 b 是小写字母（不一定不同），表示单字母变量名。
    只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，否则返回 false。 
    """
    def equationsPossible(self, equations: List[str]) -> bool:
        """
        根据查并集，等式相等表示联通，不等表示未连通
        :param equations:
        :return:
        """
        uf = UF(26)
        for idx, eqt in enumerate(equations):
            if(eqt[1] == '='):
                uf.union(ord(eqt[0])-97, ord(eqt[3])-97)

        for idx, eqt in enumerate(equations):
            if(eqt[1] == '!'):
                if(uf.connected(ord(eqt[0])-97, ord(eqt[3])-97)):
                    return False
        return True
    """
    739. 每日温度
    请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。
    如果气温在这之后都不会升高，请在该位置用 0 来代替。
    例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。
    提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。
    """
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        """
        单调栈
        :param T:
        :return:
        """
        n = len(T)
        res = [0 for _ in range(n)]
        stack = []

        for i in range(n-1, -1, -1):
            while(stack and T[stack[-1]] <= T[i]):
                stack.pop()
            res[i] = (stack[-1] - i) if(stack) else 0
            stack.append(i)
        return res
    """
    1011. 在 D 天内送达包裹的能力
    传送带上的包裹必须在 D 天内从一个港口运送到另一个港口。
    传送带上的第 i 个包裹的重量为 weights[i]。
    每一天，我们都会按给出重量的顺序往传送带上装载包裹。我们装载的重量不会超过船的最大运载重量。
    返回能在 D 天内将传送带上的所有包裹送达的船的最低运载能力。
    """
    def canFinish(self, weights: List[int], D: int, cap: int):
        n = len(weights)
        i = 0
        for d in range(D):
            _cap = cap
            while((_cap := _cap - weights[i]) >= 0):
                i += 1
                if(i >= n):
                    return True
        return False

    def shipWithinDays(self, weights: List[int], D: int) -> int:
        """
        运载能力的范围[max(weights), sum(weights)], 1 =< D <= len(weights)
        :param weights:
        :param D:
        :return:
        """
        left = max(weights)
        right = sum(weights) + 1

        while(left < right):
            mid = left + (right - left)//2
            # 能完成
            if(self.canFinish(weights, D, mid)):
                right = mid
            else:
                left = mid + 1
        return left
    """
    141. 环形链表
    给定一个链表，判断链表中是否有环。
    如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，
    我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。
    注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
    如果链表中存在环，则返回 true 。 否则，返回 false 。

    进阶：
    你能用 O(1)（即，常量）内存解决此问题吗？
    """
    def hasCycle(self, head: ListNode) -> bool:
        """
        快慢指针
        :param head:
        :return:
        """
        slow = head
        fast = head

        while(fast and fast.next):
            # 快指针走两步
            fast = fast.next.next
            # 慢指针走一步
            slow = slow.next
            if(slow == fast):
                return True
        return False

    """
    142. 环形链表 II
    给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
    为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 
    如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
    
    说明：不允许修改给定的链表。
    
    进阶：
    你是否可以使用 O(1) 空间解决此题？
    """
    def detectCycle(self, head: ListNode) -> ListNode:
        """
        快慢指针
        :param head:
        :return:
        """
        slow = head
        fast = head

        while(fast and fast.next):
            fast = fast.next.next
            slow = slow.next
            if(slow == fast):
                break
        # 无环链表
        if(not fast or not fast.next):
            return None
        """
        快指针走过的节点数(2k)是慢指针的两倍(K)
        相遇点到环起点的距离为M
        慢指针到环起点走过的距离 K - M
        快指针到环起点走做的距离 2K - M
        2K - M - (K-M) = K 是环节点的整数倍
        则链表起点到环的起点的距离为 K-M
        慢指针从链表起点走过 K - M，到达环起点
        快指针从相遇点M，走过 K - M ，也到达起点 
        """
        slow = head

        while(slow != fast):
            slow = slow.next
            fast = fast.next
        return slow

    """
    19. 删除链表的倒数第N个节点
    给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。
    """
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        """
        快慢指针
        :param head:
        :param n:
        :return:
        """
        slow = head
        fast = head

        while(n > 0):
            n -= 1
            fast = fast.next
        # 链表的长度刚好为n
        if(not fast):
            return  head.next
        # slow与fast的间距始终为n，fast到达终点时，slow.next就是倒数第n个节点
        while(fast and fast.next):
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return head
    """
    494. 目标和
    给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符号 + 和 -。对于数组中的任意一个整数，
    你都可以从 + 或 -中选择一个符号添加在前面。

    返回可以使最终数组和为目标数 S 的所有添加符号的方法数。
    """
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        """
        将数组nums分为两部分，A（+）和B（-），
        sum(A) - sum(B) = target
        sum(A) = target + sum(B)
        sum(A) + sum(A) = target + sum(B) + sum(A)
        2 * sum(A) = target + sum(nums)
        sum(A) = (target + sum(nums)) // 2
        转化为背包问题, 背包容量为sum(A), 从nums数组中选择，刚好装满背包
        :param nums:
        :param S:
        :return:
        """
        _sum = sum(nums)
        if(_sum < S or (_sum + S)%2 == 1):
            return 0

        target = int((S + sum(nums))/2)

        n = len(nums)
        dp = [[0 for _ in range(target + 1)] for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = 1

        for i in range(1, n + 1):
            for j in range(0, target + 1):
                if(j >= nums[i-1]):
                    dp[i][j] = dp[i-1][j] + dp[i-1][j - nums[i-1]]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[n][target]

    """
    1143. 最长公共子序列
    给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
    一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
    例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。
    若这两个字符串没有公共子序列，则返回 0。
    """
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1 = len(text1)
        n2 = len(text2)

        if(n1 == 0 or n2 == 0):
            return 0

        # dp数组，dp[i][j]表示 text1[...i]与text2[...j]之间的最长公共字串的长度
        # 边界条件 dp[0][j] = dp[i][0] = 0
        dp = [[0 for _ in range(n2+1)] for _ in range(n1+1)]

        for i in range(1, n1+1):
            for j in range(1, n2+1):
                # 字符相同，说明 text1[i-1] 和 text2[j-1] 都在子序列中
                if(text1[i-1] == text2[j-1]):
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[n1][n2]
    """
    583. 两个字符串的删除操作
    给定两个单词 word1 和 word2，找到使得 word1 和 word2 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。
    """
    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        n = self.longestCommonSubsequence(word1, word2)
        res = n1 + n2 - 2 * n
        return res
    """
    712. 两个字符串的最小ASCII删除和
    给定两个字符串s1, s2，找到使两个字符串相等所需删除字符的ASCII值的最小和。
    """
    def sumStrASCII(self, s: str, start: int, end: int):
        n = len(s)
        if(start >= n):
            return 0
        if(end > n):
            end = n
        if(start > end):
            return 0
        res = 0
        for i in range(start, end):
            res += ord(s[i])
        return res

    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        n1 = len(s1)
        n2 = len(s2)
        # dp数组，dp[i][j]表示 text1[...i]与text2[...j]之间需要移除的最少字符的ASCII码的和
        dp = [[-1 for _ in range(n2+1)] for _ in range(n1+1)]
        # s1 s2 均为空，则不需要任何删除
        dp[0][0] = 0
        # 边界条件 dp[0][j] = sum(s2[...j]) dp[i][0] = sum(s1[i])
        for i in range(1, n1+1):
            dp[i][0] = self.sumStrASCII(s1, 0, i)

        for j in range(1, n2+1):
            dp[0][j] = self.sumStrASCII(s2, 0, j)

        for i in range(1, n1+1):
            for j in range(1, n2+1):
                # 字符相等，则这两个字符都不需要移除，此时需要移除的最小ASCII码和等于上一次
                if(s1[i-1] == s2[j-1]):
                    dp[i][j] = dp[i-1][j-1]
                else:
                # 两个字符不相等，则要移除至少移除一个
                    dp[i][j] = min(ord(s1[i-1]) + dp[i-1][j], ord(s2[j-1]) + dp[i][j-1])
        return dp[n1][n2]
    """
    452. 用最少数量的箭引爆气球
    在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。
    由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。
    一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 
    且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。
    我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。
    给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。
    """
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        """
        等价于去掉重叠的子区间，找到，不重叠区间的个数
        :param points:
        :return:
        """
        n = len(points)
        if (n == 0):
            return 0

        # 排序，按终点升序排列
        def compareRule(a, b):
            return a[1] - b[1]

        points.sort(key=functools.cmp_to_key(compareRule))

        count = 1
        xEnd = points[0][1]
        for i in range(1, n):
            start = points[i][0]
            # 取等号时，区间边界相邻，也算区间重合
            if (start > xEnd):
                count += 1
                xEnd = points[i][1]
        return count

    """
    312. 戳气球
    有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。
    现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。
    这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。
    求所能获得硬币的最大数量。
    """
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        if(n == 0):
            return 0

        # 凑结构，使子问题相互独立
        # 构建新的数组points,其中，points[0] = points[n+1] = 1
        # pints[1...n] = nums[0...n-1]
        points = [0 for _ in range(n+2)]
        points[0] = points[n + 1] = 1
        for i in range(1, n+1):
            points[i] = nums[i-1]

        # 定义dp数组， dp[i][j]表示points[i....j]（不包括i和j）之间能获得的最大硬币数
        # 边界条件：当i >= j时，dp[i][j] = 0
        dp = [[0 for _ in range(n+2)] for _ in range(n+2)]

        #按dp表，从下到上，从左到右进行遍历
        # 从下到上
        for i in range(n, -1, -1):
            # 从左到右, j > i
            for j in range(i + 1, n+2):
                # k 为 （i,j）之间，最后一个 *被戳破* 的气球
                # 遍历所有可能的k
                for k in range(i+1, j):
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + points[i]*points[k]*points[j])
        return dp[0][n+1]
    """
    28. 实现 strStr()
    实现 strStr() 函数。
    给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。
    如果不存在，则返回  -1。
    """
    def strStr(self, haystack: str, needle: str) -> int:
        N = len(haystack)
        M = len(needle)
        if (M == 0):
            return 0
        for i in range(N - M + 1):
            j = 0
            matched = True
            for j in range(M):

                if (haystack[i + j] != needle[j]):
                    matched = False
                    break
            if (matched and j + 1 == M):
                return i
        return -1
    def strStrII(self, haystack: str, needle: str):
        kmp = KMP(needle)
        return kmp.search(haystack)
    """
    78. 子集
    给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
    解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
    """
    def subsetsRevurrent(self, nums: List[int]) -> List[List[int]]:
        if (not nums):
            return [[]]
        last = nums.pop()
        res = self.subsets(nums)
        l = len(res)
        for i in range(l):
            s = res[i].copy()
            s.append(last)
            res.append(s)
        return res
    def subsetsRecall(self, nums: List[int]) -> List[List[int]]:
        trace = []
        res = []
        n = len(nums)

        def recall(nums: List[int], start: int, trace: List[int]):
            res.append(trace.copy())

            for i in range(start, n):
                trace.append(nums[i])
                recall(nums, i + 1, trace)
                trace.remove(nums[i])

        recall(nums, 0, trace)
        return res
    """
    77. 组合
    给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
    """
    def combine(self, n: int, k: int) -> List[List[int]]:
        # trace 保存路径
        trace = []
        res = []
        if (k > n or k <= 0 or n <= 0):
            return []

        def recall(n, s, k, trace):
            if (len(trace) == k):
                res.append(trace.copy())
                return
            for i in range(s, n + 1):
                trace.append(i)
                recall(n, i + 1, k, trace)
                trace.remove(i)

        recall(n, 1, k, trace)
        return res

    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
