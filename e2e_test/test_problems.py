# Test problems for e2e validation

TEST_PROBLEM = {
    "id": "two_sum",
    "title": "Two Sum",
    "difficulty": "easy",
    "statement": """
Given an array of integers nums and an integer target, return indices of the two
numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not
use the same element twice.

You can return the answer in any order.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]

Constraints:
- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
- Only one valid answer exists.
""",
    "test_cases": [
        {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected": [0, 1]},
        {"input": {"nums": [3, 2, 4], "target": 6}, "expected": [1, 2]},
        {"input": {"nums": [3, 3], "target": 6}, "expected": [0, 1]},
        {"input": {"nums": [1, 5, 3, 7, 2], "target": 8}, "expected": [1, 2]},
    ],
    "function_name": "twoSum",
    "function_signature": "def twoSum(nums: list[int], target: int) -> list[int]:",
}

# Additional test problems for broader coverage
VALID_PARENTHESES = {
    "id": "valid_parentheses",
    "title": "Valid Parentheses",
    "difficulty": "easy",
    "statement": """
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']',
determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "()[]{}"
Output: true

Example 3:
Input: s = "(]"
Output: false

Example 4:
Input: s = "([])"
Output: true
""",
    "test_cases": [
        {"input": {"s": "()"}, "expected": True},
        {"input": {"s": "()[]{}"}, "expected": True},
        {"input": {"s": "(]"}, "expected": False},
        {"input": {"s": "([])"}, "expected": True},
        {"input": {"s": "([)]"}, "expected": False},
        {"input": {"s": ""}, "expected": True},
    ],
    "function_name": "isValid",
    "function_signature": "def isValid(s: str) -> bool:",
}

# hi_moe-rj4: Additional problems for base model validation
REVERSE_STRING = {
    "id": "reverse_string",
    "title": "Reverse String",
    "difficulty": "easy",
    "statement": """
Write a function that reverses a string. The input string is given as an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.

Example 1:
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Example 2:
Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]
""",
    "test_cases": [
        {"input": {"s": ["h", "e", "l", "l", "o"]}, "expected": ["o", "l", "l", "e", "h"]},
        {"input": {"s": ["H", "a", "n", "n", "a", "h"]}, "expected": ["h", "a", "n", "n", "a", "H"]},
        {"input": {"s": ["a"]}, "expected": ["a"]},
        {"input": {"s": ["a", "b"]}, "expected": ["b", "a"]},
    ],
    "function_name": "reverseString",
    "function_signature": "def reverseString(s: list[str]) -> list[str]:",
}

PALINDROME_NUMBER = {
    "id": "palindrome_number",
    "title": "Palindrome Number",
    "difficulty": "easy",
    "statement": """
Given an integer x, return true if x is a palindrome, and false otherwise.

An integer is a palindrome when it reads the same forward and backward.

Example 1:
Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.

Example 2:
Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-.

Example 3:
Input: x = 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
""",
    "test_cases": [
        {"input": {"x": 121}, "expected": True},
        {"input": {"x": -121}, "expected": False},
        {"input": {"x": 10}, "expected": False},
        {"input": {"x": 0}, "expected": True},
        {"input": {"x": 12321}, "expected": True},
    ],
    "function_name": "isPalindrome",
    "function_signature": "def isPalindrome(x: int) -> bool:",
}

FIZZBUZZ = {
    "id": "fizzbuzz",
    "title": "Fizz Buzz",
    "difficulty": "easy",
    "statement": """
Given an integer n, return a string array answer (1-indexed) where:

- answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
- answer[i] == "Fizz" if i is divisible by 3.
- answer[i] == "Buzz" if i is divisible by 5.
- answer[i] == i (as a string) if none of the above conditions are true.

Example 1:
Input: n = 3
Output: ["1","2","Fizz"]

Example 2:
Input: n = 5
Output: ["1","2","Fizz","4","Buzz"]

Example 3:
Input: n = 15
Output: ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
""",
    "test_cases": [
        {"input": {"n": 3}, "expected": ["1", "2", "Fizz"]},
        {"input": {"n": 5}, "expected": ["1", "2", "Fizz", "4", "Buzz"]},
        {"input": {"n": 15}, "expected": ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]},
    ],
    "function_name": "fizzBuzz",
    "function_signature": "def fizzBuzz(n: int) -> list[str]:",
}

MERGE_SORTED_ARRAYS = {
    "id": "merge_sorted_arrays",
    "title": "Merge Sorted Array",
    "difficulty": "easy",
    "statement": """
You are given two integer arrays nums1 and nums2, sorted in non-decreasing order.

Merge nums2 into nums1 as one sorted array and return the result.

The final sorted array should not be returned by the function, but instead be
stored inside the array nums1. To accommodate this, nums1 has a length of m + n,
where the first m elements denote the elements that should be merged, and the
last n elements are set to 0 and should be ignored. nums2 has a length of n.

Example 1:
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

Example 2:
Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]

Example 3:
Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
""",
    "test_cases": [
        {"input": {"nums1": [1, 2, 3, 0, 0, 0], "m": 3, "nums2": [2, 5, 6], "n": 3}, "expected": [1, 2, 2, 3, 5, 6]},
        {"input": {"nums1": [1], "m": 1, "nums2": [], "n": 0}, "expected": [1]},
        {"input": {"nums1": [0], "m": 0, "nums2": [1], "n": 1}, "expected": [1]},
    ],
    "function_name": "merge",
    "function_signature": "def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> list[int]:",
}

MAXIMUM_SUBARRAY = {
    "id": "maximum_subarray",
    "title": "Maximum Subarray",
    "difficulty": "medium",
    "statement": """
Given an integer array nums, find the subarray with the largest sum, and return its sum.

A subarray is a contiguous non-empty sequence of elements within an array.

Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.

Example 2:
Input: nums = [1]
Output: 1
Explanation: The subarray [1] has the largest sum 1.

Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
""",
    "test_cases": [
        {"input": {"nums": [-2, 1, -3, 4, -1, 2, 1, -5, 4]}, "expected": 6},
        {"input": {"nums": [1]}, "expected": 1},
        {"input": {"nums": [5, 4, -1, 7, 8]}, "expected": 23},
        {"input": {"nums": [-1]}, "expected": -1},
    ],
    "function_name": "maxSubArray",
    "function_signature": "def maxSubArray(nums: list[int]) -> int:",
}

CLIMBING_STAIRS = {
    "id": "climbing_stairs",
    "title": "Climbing Stairs",
    "difficulty": "easy",
    "statement": """
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Example 1:
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Example 2:
Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
""",
    "test_cases": [
        {"input": {"n": 2}, "expected": 2},
        {"input": {"n": 3}, "expected": 3},
        {"input": {"n": 4}, "expected": 5},
        {"input": {"n": 5}, "expected": 8},
        {"input": {"n": 1}, "expected": 1},
    ],
    "function_name": "climbStairs",
    "function_signature": "def climbStairs(n: int) -> int:",
}

# Backward compatibility alias
MEDIUM_PROBLEM = VALID_PARENTHESES

# All problems for validation testing
ALL_PROBLEMS = [
    TEST_PROBLEM,
    VALID_PARENTHESES,
    REVERSE_STRING,
    PALINDROME_NUMBER,
    FIZZBUZZ,
    MERGE_SORTED_ARRAYS,
    MAXIMUM_SUBARRAY,
    CLIMBING_STAIRS,
]

# Categorized for targeted testing
EASY_PROBLEMS = [TEST_PROBLEM, VALID_PARENTHESES, REVERSE_STRING, PALINDROME_NUMBER, FIZZBUZZ, CLIMBING_STAIRS]
MEDIUM_PROBLEMS = [MERGE_SORTED_ARRAYS, MAXIMUM_SUBARRAY]
