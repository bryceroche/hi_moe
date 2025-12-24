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
MEDIUM_PROBLEM = {
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

ALL_PROBLEMS = [TEST_PROBLEM, MEDIUM_PROBLEM]
