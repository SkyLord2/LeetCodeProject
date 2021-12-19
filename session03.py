# -*- coding: utf-8 -*-
# @Time : 2021/12/12 11:01
# @Author : cds
# @Site : https://github.com/SkyLord2?tab=repositories
# @Email: chengdongsheng@outlook.com
# @File : session03.py
# @Software: PyCharm
from typing import List
class Solution:
    def __init__(self):
        self.memo = {}
    """
    509. 斐波那契数
    斐波那契数，通常用 F(n) 表示，形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
    F(0) = 0，F(1) = 1
    F(n) = F(n - 1) + F(n - 2)，其中 n > 1
    给你 n ，请计算 F(n) 。
    """
    def fib(self, n: int) -> int:
        if (n == 0):
            return 0
        if (n == 1):
            return 1
        res = 0
        # base case
        cur = 1
        pre = 0
        # 状态转移
        for i in range(2, n + 1):
            res = pre + cur
            pre = cur
            cur = res
        return res
    """
    322. 零钱兑换
    给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
    计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
    你可以认为每种硬币的数量是无限的。
    注意：
    1 <= coins.length <= 12
    1 <= coins[i] <= 231 - 1
    0 <= amount <= 104
    """
    def coinChange(self, coins: List[int], amount: int) -> int:
        # 凑出总金额 amount 最多需要 amount 枚硬币
        dp = [10001 for i in range(amount + 1)]
        dp[0] = 0
        for i in range(1, amount + 1):
            for idx, coin in enumerate(coins):
                if (i - coin >= 0):
                    dp[i] = min(dp[i - coin] + 1, dp[i])
        return -1 if (dp[amount] == 10001) else dp[amount]