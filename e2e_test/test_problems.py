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

# Harder problems that benefit from tier hierarchy planning
HARD_PROBLEM_1 = {
    "id": "merge_intervals",
    "title": "Merge Intervals",
    "difficulty": "medium",
    "statement": """
Given an array of intervals where intervals[i] = [start_i, end_i], merge all
overlapping intervals, and return an array of the non-overlapping intervals
that cover all the intervals in the input.

To solve this problem, you should first sort the intervals by their start times.
Then, iterate through the sorted intervals and merge overlapping ones.
Finally, return the merged result.

Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.

Constraints:
- 1 <= intervals.length <= 10^4
- intervals[i].length == 2
- 0 <= start_i <= end_i <= 10^4
""",
    "test_cases": [
        {"input": {"intervals": [[1,3],[2,6],[8,10],[15,18]]}, "expected": [[1,6],[8,10],[15,18]]},
        {"input": {"intervals": [[1,4],[4,5]]}, "expected": [[1,5]]},
        {"input": {"intervals": [[1,4],[0,4]]}, "expected": [[0,4]]},
        {"input": {"intervals": [[1,4],[2,3]]}, "expected": [[1,4]]},
        {"input": {"intervals": [[1,4]]}, "expected": [[1,4]]},
    ],
    "function_name": "merge",
    "function_signature": "def merge(intervals: list[list[int]]) -> list[list[int]]:",
}

HARD_PROBLEM_2 = {
    "id": "longest_palindrome",
    "title": "Longest Palindromic Substring",
    "difficulty": "medium",
    "statement": """
Given a string s, return the longest palindromic substring in s.

A palindrome is a string that reads the same forwards and backwards.

To find the longest palindrome, you can use the expand-around-center approach:
First, iterate through each character as a potential center.
Then, expand outward from each center to find palindromes.
Finally, track and return the longest one found.

Example 1:
Input: s = "babad"
Output: "bab" or "aba" (both are valid)

Example 2:
Input: s = "cbbd"
Output: "bb"

Constraints:
- 1 <= s.length <= 1000
- s consist of only digits and English letters.
""",
    "test_cases": [
        {"input": {"s": "babad"}, "expected": "bab"},  # or "aba"
        {"input": {"s": "cbbd"}, "expected": "bb"},
        {"input": {"s": "a"}, "expected": "a"},
        {"input": {"s": "ac"}, "expected": "a"},  # or "c"
        {"input": {"s": "racecar"}, "expected": "racecar"},
    ],
    "function_name": "longestPalindrome",
    "function_signature": "def longestPalindrome(s: str) -> str:",
}

HARD_PROBLEM_3 = {
    "id": "course_schedule",
    "title": "Course Schedule",
    "difficulty": "medium",
    "statement": """
There are a total of numCourses courses you have to take, labeled from 0 to
numCourses - 1. You are given an array prerequisites where prerequisites[i] =
[a_i, b_i] indicates that you must take course b_i before course a_i.

Return true if you can finish all courses. Otherwise, return false.

This is a cycle detection problem in a directed graph. First, build an adjacency
list from the prerequisites. Then, use depth-first search to detect cycles.
Finally, return whether all courses can be completed without circular dependencies.

Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: Take course 0 first, then course 1.

Example 2:
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There's a cycle between course 0 and 1.

Constraints:
- 1 <= numCourses <= 2000
- 0 <= prerequisites.length <= 5000
- prerequisites[i].length == 2
- 0 <= a_i, b_i < numCourses
- All pairs are unique
""",
    "test_cases": [
        {"input": {"numCourses": 2, "prerequisites": [[1,0]]}, "expected": True},
        {"input": {"numCourses": 2, "prerequisites": [[1,0],[0,1]]}, "expected": False},
        {"input": {"numCourses": 3, "prerequisites": [[1,0],[2,1]]}, "expected": True},
        {"input": {"numCourses": 4, "prerequisites": [[1,0],[2,0],[3,1],[3,2]]}, "expected": True},
        {"input": {"numCourses": 1, "prerequisites": []}, "expected": True},
    ],
    "function_name": "canFinish",
    "function_signature": "def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:",
}

# Easy problems (baseline can solve, fast path applies)
EASY_PROBLEMS = [TEST_PROBLEM, MEDIUM_PROBLEM]

# Medium/Hard problems (hierarchy may have advantage, no fast path)
HARD_PROBLEMS = [HARD_PROBLEM_1, HARD_PROBLEM_2, HARD_PROBLEM_3]

ALL_PROBLEMS = EASY_PROBLEMS + HARD_PROBLEMS
