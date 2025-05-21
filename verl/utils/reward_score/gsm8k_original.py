# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def extract_solution(solution_str: str, method: str = "strict"):
    assert method in ["strict", "flexible"]

    m = re.search(r"<answer>\s*([-0-9.,]+).*?</answer>", solution_str, flags=re.I | re.S)
    if m:
        return m.group(1).replace(",", "").replace("$", "")

    # 1) 기존 #### 패턴
    if method == "strict":
        m = re.search(r"####\\s*([-0-9.,]+)", solution_str)
        return m.group(1).replace(",", "").replace("$", "") if m else None

    # 2) flexible 모드: 마지막 숫자
    nums = re.findall(r"[-0-9.,]+", solution_str)
    for num in reversed(nums):
        if num.strip().strip("."):
            return num.replace(",", "").replace("$", "")
    return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
