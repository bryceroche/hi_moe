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

# Genuinely hard problems that baseline often fails on
VERY_HARD_PROBLEM_1 = {
    "id": "lru_cache",
    "title": "LRU Cache",
    "difficulty": "hard",
    "statement": """
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:
- LRUCache(capacity) Initialize the LRU cache with positive size capacity.
- get(key) Return the value of the key if it exists, otherwise return -1.
- put(key, value) Update the value if key exists. Otherwise, add key-value pair.
  If the number of keys exceeds capacity, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.

To achieve O(1) operations, you need to combine two data structures:
First, use a hashmap for O(1) key lookup.
Then, use a doubly linked list to track recency order.
Finally, update both structures on every get/put operation.

Example:
Input: ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
       [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output: [null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation:
LRUCache lru = new LRUCache(2);
lru.put(1, 1); // cache is {1=1}
lru.put(2, 2); // cache is {1=1, 2=2}
lru.get(1);    // return 1, cache is {2=2, 1=1} (1 is now most recent)
lru.put(3, 3); // evicts key 2, cache is {1=1, 3=3}
lru.get(2);    // returns -1 (not found)
lru.put(4, 4); // evicts key 1, cache is {3=3, 4=4}
lru.get(1);    // return -1 (not found)
lru.get(3);    // return 3
lru.get(4);    // return 4

Constraints:
- 1 <= capacity <= 3000
- 0 <= key <= 10^4
- 0 <= value <= 10^5
- At most 2 * 10^5 calls to get and put
""",
    "test_cases": [
        {
            "input": {
                "operations": ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"],
                "args": [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
            },
            "expected": [None, None, None, 1, None, -1, None, -1, 3, 4]
        },
        {
            "input": {
                "operations": ["LRUCache", "put", "get", "put", "get", "get"],
                "args": [[1], [2, 1], [2], [3, 2], [2], [3]]
            },
            "expected": [None, None, 1, None, -1, 2]
        },
        {
            "input": {
                "operations": ["LRUCache", "put", "put", "put", "put", "get", "get"],
                "args": [[2], [2, 1], [1, 1], [2, 3], [4, 1], [1], [2]]
            },
            "expected": [None, None, None, None, None, -1, 3]
        },
    ],
    "function_name": "LRUCache",
    "function_signature": "class LRUCache:\n    def __init__(self, capacity: int):\n    def get(self, key: int) -> int:\n    def put(self, key: int, value: int) -> None:",
    "validation_type": "class_operations",
}

VERY_HARD_PROBLEM_2 = {
    "id": "min_window_substring",
    "title": "Minimum Window Substring",
    "difficulty": "hard",
    "statement": """
Given two strings s and t of lengths m and n respectively, return the minimum window
substring of s such that every character in t (including duplicates) is included
in the window. If there is no such substring, return the empty string "".

A substring is a contiguous sequence of characters within a string.

To solve this problem efficiently, use the sliding window technique:
First, count all characters needed from t using a frequency map.
Then, expand the right pointer to include characters until all are found.
Finally, contract from the left to find the minimum valid window.
Track the minimum window seen and return it.

Example 1:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from t.

Example 2:
Input: s = "a", t = "a"
Output: "a"

Example 3:
Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be in the window. Since s only has one 'a', impossible.

Constraints:
- 1 <= s.length, t.length <= 10^5
- s and t consist of uppercase and lowercase English letters
""",
    "test_cases": [
        {"input": {"s": "ADOBECODEBANC", "t": "ABC"}, "expected": "BANC"},
        {"input": {"s": "a", "t": "a"}, "expected": "a"},
        {"input": {"s": "a", "t": "aa"}, "expected": ""},
        {"input": {"s": "ab", "t": "b"}, "expected": "b"},
        {"input": {"s": "bba", "t": "ab"}, "expected": "ba"},
        {"input": {"s": "aaaaaaaaaaaabbbbbcdd", "t": "abcdd"}, "expected": "abbbbbcdd"},
    ],
    "function_name": "minWindow",
    "function_signature": "def minWindow(s: str, t: str) -> str:",
}

VERY_HARD_PROBLEM_3 = {
    "id": "trapping_rain_water",
    "title": "Trapping Rain Water",
    "difficulty": "hard",
    "statement": """
Given n non-negative integers representing an elevation map where the width of
each bar is 1, compute how much water it can trap after raining.

This is a classic problem with multiple solution approaches.
First, understand that water at position i is bounded by the minimum of
the maximum heights to its left and right, minus its own height.
Then, you can use two pointers moving inward to compute this efficiently.
Finally, accumulate the trapped water at each position.

Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The elevation map is shown below. The blue sections represent trapped water.
    |
    |       █
    |   █~~~█~█~█
    | █~█~█~█~█~█~█
    └─────────────
     0 1 0 2 1 0 1 3 2 1 2 1

Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9

Constraints:
- n == height.length
- 1 <= n <= 2 * 10^4
- 0 <= height[i] <= 10^5
""",
    "test_cases": [
        {"input": {"height": [0,1,0,2,1,0,1,3,2,1,2,1]}, "expected": 6},
        {"input": {"height": [4,2,0,3,2,5]}, "expected": 9},
        {"input": {"height": [1,0,1]}, "expected": 1},
        {"input": {"height": [5,4,1,2]}, "expected": 1},
        {"input": {"height": [0,0,0,0]}, "expected": 0},
        {"input": {"height": [5,2,1,2,1,5]}, "expected": 14},
    ],
    "function_name": "trap",
    "function_signature": "def trap(height: list[int]) -> int:",
}

VERY_HARD_PROBLEM_4 = {
    "id": "word_ladder",
    "title": "Word Ladder",
    "difficulty": "hard",
    "statement": """
A transformation sequence from word beginWord to word endWord using a dictionary
wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:
- Every adjacent pair of words differs by a single letter.
- Every si for 1 <= i <= k is in wordList.
- beginWord does not need to be in wordList.
- sk == endWord

Given two words beginWord and endWord, and a dictionary wordList, return the number
of words in the shortest transformation sequence from beginWord to endWord, or 0
if no such sequence exists.

To solve this efficiently, model it as a graph problem:
First, treat each word as a node connected to words differing by one letter.
Then, use BFS from beginWord to find the shortest path to endWord.
Finally, return the path length (number of words including start and end).

Example 1:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: hit -> hot -> dot -> dog -> cog

Example 2:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: 0
Explanation: endWord "cog" is not in wordList

Constraints:
- 1 <= beginWord.length <= 10
- endWord.length == beginWord.length
- 1 <= wordList.length <= 5000
- wordList[i].length == beginWord.length
- beginWord, endWord, and wordList[i] consist of lowercase English letters
- beginWord != endWord
- All words in wordList are unique
""",
    "test_cases": [
        {"input": {"beginWord": "hit", "endWord": "cog", "wordList": ["hot","dot","dog","lot","log","cog"]}, "expected": 5},
        {"input": {"beginWord": "hit", "endWord": "cog", "wordList": ["hot","dot","dog","lot","log"]}, "expected": 0},
        {"input": {"beginWord": "a", "endWord": "c", "wordList": ["a","b","c"]}, "expected": 2},
        {"input": {"beginWord": "hot", "endWord": "dog", "wordList": ["hot","dog","dot"]}, "expected": 3},
        {"input": {"beginWord": "game", "endWord": "thee", "wordList": ["frye","heat","tree","thee","game","free","hell","fame","faye"]}, "expected": 0},
    ],
    "function_name": "ladderLength",
    "function_signature": "def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:",
}

VERY_HARD_PROBLEM_5 = {
    "id": "median_two_sorted_arrays",
    "title": "Median of Two Sorted Arrays",
    "difficulty": "hard",
    "statement": """
Given two sorted arrays nums1 and nums2 of size m and n respectively, return
the median of the two sorted arrays.

The overall run time complexity should be O(log(m+n)).

This problem requires binary search on the partition point:
First, ensure you're searching on the smaller array for efficiency.
Then, binary search to find a partition where all left elements <= all right elements.
Finally, compute the median from the partition boundary elements.

Example 1:
Input: nums1 = [1,3], nums2 = [2]
Output: 2.0
Explanation: merged = [1,2,3], median = 2.0

Example 2:
Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.5
Explanation: merged = [1,2,3,4], median = (2+3)/2 = 2.5

Example 3:
Input: nums1 = [], nums2 = [1]
Output: 1.0

Constraints:
- nums1.length == m
- nums2.length == n
- 0 <= m, n <= 1000
- 1 <= m + n <= 2000
- -10^6 <= nums1[i], nums2[i] <= 10^6
""",
    "test_cases": [
        {"input": {"nums1": [1,3], "nums2": [2]}, "expected": 2.0},
        {"input": {"nums1": [1,2], "nums2": [3,4]}, "expected": 2.5},
        {"input": {"nums1": [], "nums2": [1]}, "expected": 1.0},
        {"input": {"nums1": [2], "nums2": []}, "expected": 2.0},
        {"input": {"nums1": [1,2,3,4,5], "nums2": [6,7,8,9,10]}, "expected": 5.5},
        {"input": {"nums1": [1,1,1,1], "nums2": [1,1,1,1]}, "expected": 1.0},
    ],
    "function_name": "findMedianSortedArrays",
    "function_signature": "def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:",
}

# Easy problems (baseline can solve, fast path applies)
EASY_PROBLEMS = [TEST_PROBLEM, MEDIUM_PROBLEM]

# Medium problems (classics that baseline likely knows)
MEDIUM_PROBLEMS = [HARD_PROBLEM_1, HARD_PROBLEM_2, HARD_PROBLEM_3]

# Hard problems (genuinely challenging, baseline often fails)
HARD_PROBLEMS = [VERY_HARD_PROBLEM_2, VERY_HARD_PROBLEM_3, VERY_HARD_PROBLEM_4, VERY_HARD_PROBLEM_5]

# All problems for full benchmark
ALL_PROBLEMS = EASY_PROBLEMS + MEDIUM_PROBLEMS + HARD_PROBLEMS
