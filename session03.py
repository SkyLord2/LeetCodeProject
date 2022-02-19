# -*- coding: utf-8 -*-
# @Time : 2021/12/12 11:01
# @Author : cds
# @Site : https://github.com/SkyLord2?tab=repositories
# @Email: chengdongsheng@outlook.com
# @File : session03.py
# @Software: PyCharm
from typing import List
import functools
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
    """
    300. 最长递增子序列
    给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
    子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。
    例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
    提示：
        1 <= nums.length <= 2500
        -104 <= nums[i] <= 104
    进阶：
        你可以设计时间复杂度为 O(n2) 的解决方案吗？
        你能将算法的时间复杂度降低到 O(nlog(n)) 吗?
    """
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        # 前 i 个元素的最长子序列的长度
        dp = [1 for i in range(n)]
        # O(n2)
        for i in range(0, n):
            for j in range(i):
                if(nums[i] > nums[j]):
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
    """
    patient sort(耐心排序)
    只能把点数小的牌压到点数比它大的牌上；如果当前牌点数较大没有可以放置的堆，则新建一个堆，把这张牌放进去；如果当前牌有多个堆可供选择，则选择最左边的那一堆放置。
    """
    def lengthOfLIS_BS(self, nums: List[int]) -> int:
        n = len(nums)
        # 记录顶部的数值
        top = [-200 for i in range(n)]
        piles = 0

        for (idx, item) in enumerate(nums):
            left = 0
            right = piles
            while(left < right):
                mid = left + (right - left)//2
                topVal = top[mid]
                if(item > topVal):
                    left = mid + 1
                elif(item < topVal):
                    right = mid
                else:
                    right = mid
            if(left == piles):
                piles += 1
            top[left] = item
    """
    931. 下降路径最小和
    给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。

    下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素。
    在下一行选择的元素和当前行所选元素最多相隔一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。
    具体来说，位置 (row, col) 的下一个元素应当是 (row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。
    """
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix)
        dp = [[0 for i in range(n)] for j in range(n)]
        for (idx, item) in enumerate(matrix[0]):
            dp[0][idx] = item

        for i in range(1, n):
            for j in range(n):
                cur = matrix[i][j]
                c = []
                c.append(dp[i-1][j] + cur)
                if(j - 1 >= 0):
                    c.append(dp[i-1][j-1] + cur)
                if(j + 1 <= n-1):
                    c.append(dp[i-1][j+1] + cur)
                dp[i][j] = min(c)
        return min(dp[n-1])
    """
    72. 编辑距离
    给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数。
    你可以对一个单词进行如下三种操作：
    插入一个字符
    删除一个字符
    替换一个字符
    
    提示：
    0 <= word1.length, word2.length <= 500
    word1 和 word2 由小写英文字母组成
    """
    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        if (n1 == 0):
            return n2
        if (n2 == 0):
            return n1
        # dp 数组 dp[i][j]表示 word1[0 ~ i-1]前i个字母 word2[0 ~ j-1]前j个字母之间的最小编辑距离
        dp = [[0 for j in range(n2 + 1)] for i in range(n1 + 1)]
        # base case
        for i in range(1, n1 + 1):
            dp[i][0] = i
        for j in range(1, n2 + 1):
            dp[0][j] = j
        # 状态转移
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if (word1[i - 1] == word2[j - 1]):
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # dp[i - 1][j] + 1          插入操作
                    # dp[i][j - 1] + 1          移除操作
                    # dp[i - 1][j - 1] + 1      替换操作
                    dp[i][j] = min([dp[i - 1][j - 1] + 1, dp[i - 1][j] + 1, dp[i][j - 1] + 1])
        return dp[n1][n2]
    """
    354. 俄罗斯套娃信封问题
    给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。
    当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
    请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。
    
    注意：不允许旋转信封。
    提示：
        1 <= envelopes.length <= 5000
        envelopes[i].length == 2
        1 <= wi, hi <= 10^4
    """
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # 排序，按宽度升序排列，宽度相同，高度降序排列，固定住 宽度，只处理 高度
        def compareRule(a, b):
            if (a[0] == b[0]):
                return b[1] - a[1]
            return a[0] - b[0]
        envelopes.sort(key=functools.cmp_to_key(compareRule))
        n = len(envelopes)
        hs = []
        for i in range(n):
            it = envelopes[i]
            hs.append(it[1])
        # 最少 嵌套一个
        dp = [1 for i in range(n)]

        for i in range(n):
            for j in range(i):
                if(hs[i] > hs[j]):
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)