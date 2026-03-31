import contextlib

import cv2
import torch

from PIL import Image


MAX_NEW_TOKENS = 16
NUM_BEAMS = 2


def _to_pil_images(images):
    pil_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(image_rgb))
    return pil_images


def _generate_outputs(pil_images, model, processor, device):
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    with torch.inference_mode():
        with autocast_context:
            return model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )


def recognize_numbers(images, model, processor, device):
    """Read digits from one or more cropped cell images.

    Returns:
        list[tuple[predicted_string, confidence_score]]
        confidence_score is the minimum token probability in the generated sequence.
    """

    if not images:
        return []

    pil_images = _to_pil_images(images)
    outputs = _generate_outputs(pil_images, model, processor, device)

    sequences = outputs.sequences
    beam_indices = getattr(outputs, "beam_indices", None)
    special_token_ids = {
        token_id
        for token_id in (
            processor.tokenizer.bos_token_id,
            processor.tokenizer.eos_token_id,
            processor.tokenizer.pad_token_id,
        )
        if token_id is not None
    }
    min_confidences = []
    for batch_idx in range(sequences.shape[0]):
        min_confidence = 1.0
        for step in range(sequences.shape[1] - 1):
            if step >= len(outputs.scores):
                break

            token_id = int(sequences[batch_idx, step + 1].item())
            if token_id in special_token_ids:
                continue

            if beam_indices is None:
                beam_row = batch_idx
            else:
                beam_row = int(beam_indices[batch_idx, step].item())
                if beam_row < 0:
                    continue

            probs = torch.softmax(outputs.scores[step][beam_row], dim=-1)
            token_prob = probs[token_id].item()
            min_confidence = min(min_confidence, token_prob)

        min_confidences.append(min_confidence)

    raw_predictions = processor.batch_decode(outputs.sequences, skip_special_tokens=True)

    results = []
    for raw_prediction, min_confidence in zip(raw_predictions, min_confidences):
        digits = "".join(ch for ch in raw_prediction if ch.isdigit())
        if not digits:
            digits = "?"
            min_confidence = 0.0

        print(f"[OCR] Digits: '{digits}' Confidence: {min_confidence:.2%}")
        results.append((digits, float(min_confidence)))

    return results


def recognize_number(image, model, processor, device):
    """Backward-compatible single-image OCR wrapper."""

    return recognize_numbers([image], model, processor, device)[0]
