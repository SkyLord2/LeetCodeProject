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
            nums.sort(reverse=False)
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
        pass