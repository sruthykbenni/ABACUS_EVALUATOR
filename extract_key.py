# pipeline/extract_key.py

import fitz
import re


def is_arithmetic_progression(nums):
    if len(nums) < 3:
        return False

    step = nums[1] - nums[0]
    if step <= 0:
        return False

    return all(nums[i + 1] - nums[i] == step for i in range(len(nums) - 1))


def looks_like_question_blocks(blocks):
    if not blocks:
        return False

    flat = [num for block in blocks for num in block["nums"]]
    if not flat or min(flat) != 1:
        return False

    unique = sorted(set(flat))
    return len(flat) == len(unique) and unique == list(range(1, unique[-1] + 1))


def extract_numeric_blocks(page):
    numeric_blocks = []
    for block in page.get_text("blocks"):
        nums = list(map(int, re.findall(r"\b\d+\b", block[4])))
        if len(nums) < 3:
            continue

        numeric_blocks.append(
            {
                "x0": block[0],
                "y0": block[1],
                "nums": nums,
            }
        )

    return numeric_blocks


def extract_answer_key(pdf_path):
    doc = fitz.open(pdf_path)

    full_key = {}

    for page_index, page in enumerate(doc):
        page_text = page.get_text().strip()
        if not page_text and page.get_images(full=True):
            raise ValueError(
                f"Page {page_index+1}: answer key PDF is image-only. "
                "This extractor supports text-based answer-key PDFs only."
            )

        numeric_blocks = extract_numeric_blocks(page)
        question_blocks = [
            block for block in numeric_blocks if is_arithmetic_progression(block["nums"])
        ]
        answer_blocks = numeric_blocks

        page_answers = {}

        if looks_like_question_blocks(question_blocks):
            answer_blocks = [
                block
                for block in numeric_blocks
                if not is_arithmetic_progression(block["nums"])
            ]

            question_rows = sorted(question_blocks, key=lambda block: (block["y0"], block["x0"]))
            answer_rows = sorted(answer_blocks, key=lambda block: (block["y0"], block["x0"]))

            if len(question_rows) == len(answer_rows) and all(
                len(q_block["nums"]) == len(a_block["nums"])
                for q_block, a_block in zip(question_rows, answer_rows)
            ):
                for q_block, a_block in zip(question_rows, answer_rows):
                    for question, answer in zip(q_block["nums"], a_block["nums"]):
                        page_answers[question] = str(answer)

        if not page_answers:
            section_answers = []
            for block in answer_blocks:
                section_answers.extend(block["nums"])

            page_answers = {i + 1: str(ans) for i, ans in enumerate(section_answers)}

        if not page_answers:
            raise ValueError(f"Page {page_index+1}: No answers detected")

        full_key[page_index + 1] = page_answers

    return full_key


