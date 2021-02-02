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

class Solution:
    def __init__(self):
        pass

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