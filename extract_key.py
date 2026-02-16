# pipeline/extract_key.py

import fitz
import re

def is_sequential(nums):
    if len(nums) < 3:
        return False
    return all(nums[i+1] == nums[i] + 1 for i in range(len(nums)-1))


def extract_answer_key(pdf_path):
    doc = fitz.open(pdf_path)

    full_key = {}

    for page_index, page in enumerate(doc):
        blocks = page.get_text("blocks")

        section_answers = []

        for b in blocks:
            text = b[4]
            nums = list(map(int, re.findall(r"\b\d+\b", text)))

            # Skip small blocks (headers, noise)
            if len(nums) < 5:
                continue

            # Skip question number grids (sequential)
            if is_sequential(nums):
                continue

            # Remaining numeric blocks are answer blocks
            section_answers.extend(nums)

        if not section_answers:
            raise ValueError(f"Page {page_index+1}: No answers detected")

        # Store section dynamically
        full_key[page_index + 1] = {
            i+1: str(ans) for i, ans in enumerate(section_answers)
        }

    return full_key


