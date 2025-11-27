import os
import ast
import re
import numpy as np
from PIL import Image, ImageDraw
import os.path as osp
from loguru import logger as eval_logger
from typing import Dict, List, Tuple

# Default data root from YAML configuration, can be overridden by environment variable
DEFAULT_DATA_ROOT = "/home/jensen/remote_jensen/huangjianxin/vlm_benchmark/Where2Place"
DATA_ROOT = os.getenv("DATA_ROOT", DEFAULT_DATA_ROOT)
model_id = os.getenv("MODEL_ID", "")

def draw_result(gt: Dict, mask_img: Image, score: float, points: List[Tuple[int, int]]):
    """
    Draws the result of a prediction on an image, including a mask overlay, points, and a score.

    Parameters:
        gt (Dict): Ground truth data containing metadata such as the image path and question ID.
        mask_img (Image): Binary mask image indicating regions of interest.
        score (float): Prediction score to display on the image.
        points (List[Tuple[int, int]]): List of (x, y) coordinates to mark on the image.

    Side Effects:
        Saves the resulting image with overlays and annotations to the 'output/imgs' directory.
    """
    try:
        # For debug
        # Load the original image
        img_path = gt.get("image")
        # If image is loaded as PIL object in dataset, use it, else load from path
        if isinstance(img_path, Image.Image):
             img = img_path.copy()
        elif isinstance(img_path, str):
             full_path = osp.join(gt.get("data_root", DATA_ROOT), img_path)
             if not osp.exists(full_path) and osp.exists(img_path):
                 full_path = img_path
             img = Image.open(full_path)
        else:
             # Fallback if image is not available
             return

        img = img.convert("RGBA")

        # Convert mask to numpy array and create overlay
        mask_array = np.array(mask_img)
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]  # Take first channel if RGB
        mask_array = mask_array / 255.0  # Normalize to 0-1

        # Create semi-transparent green overlay for mask
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_array = np.array(overlay)

        # Apply green color where mask is positive
        # Resize mask to match image if needed (robustness check)
        if mask_array.shape[:2] != (img.height, img.width):
             # You might want to resize mask here, but let's skip overlay if mismatch for safety
             pass
        else:
            mask_indices = mask_array > 0.5
            overlay_array[mask_indices] = [0, 255, 0, 100]  # Semi-transparent green

        overlay = Image.fromarray(overlay_array)
        img = Image.alpha_composite(img, overlay)

        # Convert back to RGB for drawing
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw points as red circles
        points_array = np.array(points)
        for point in points_array:
            x, y = int(point[0]), int(point[1])
            radius = 3
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill="red",
                outline="darkred",
            )

        score_text = f"Score: {score:.3f}"
        text_bbox = draw.textbbox((0, 0), score_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position text in top-left corner with padding
        text_x, text_y = 10, 10

        # Draw text background
        draw.rectangle(
            [text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5],
            fill="white",
            outline="black",
        )

        # Draw text
        draw.text((text_x, text_y), score_text, fill="black")

        output_dir = "output/imgs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use question_id or some unique identifier
        file_name = f"{gt.get('question_id', 'unknown')}.png"
        img.save(osp.join(output_dir, file_name))
    except Exception as e:
        eval_logger.warning(f"Failed to draw result visualization: {e}")

def where2place_doc_to_visual(doc):
    """
    Extract visual content from doc for models that use doc_to_visual interface.
    Returns a list of PIL Images.
    """
    image = doc.get("image")
    if image is None:
        return []
    
    # If image is already a PIL Image, return it
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    
    # If image is a path string, try to load it
    if isinstance(image, str):
        full_image_path = osp.join(DATA_ROOT, image)
        if osp.exists(full_image_path):
             return [Image.open(full_image_path).convert("RGB")]
        else:
             # If path doesn't exist, maybe it's already absolute or relative to cwd
             if osp.exists(image):
                  return [Image.open(image).convert("RGB")]
    
    return []

def where2place_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    image = doc["image"]
    # If image is a path string, try to load it
    if isinstance(image, str):
        full_image_path = osp.join(DATA_ROOT, image)
        if osp.exists(full_image_path):
             image = Image.open(full_image_path).convert("RGB")
        else:
             # If path doesn't exist, maybe it's already absolute or relative to cwd
             if osp.exists(image):
                  image = Image.open(image).convert("RGB")
    
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": f"{pre_prompt}{doc['question']}{post_prompt}"},
            ],
        }
    ]

def where2place_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"

def where2place_doc_to_target(doc):
    # Return the doc itself as it contains the ground truth info (mask_path, etc.)
    return doc

def parse_points(text):
    """
    Parse a list of tuples/lists from a string, e.g., "[(10, 20), [30, 40]]"
    """
    points = []
    
    # 1. Try parsing as a structured list first
    try:
        # Attempt to find the list part in the text
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            list_str = text[start:end+1]
            parsed = ast.literal_eval(list_str)
            
            if isinstance(parsed, list):
                for p in parsed:
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        # Ensure coordinates are numeric
                        if isinstance(p[0], (int, float)) and isinstance(p[1], (int, float)):
                            points.append((float(p[0]), float(p[1])))
    except Exception:
        pass
        
    # 2. If no points found, fallback to regex pattern matching (like in evaluation.py)
    if not points:
        # Regex pattern to match points like [123, 456], (123.5, 456), etc.
        point_pattern = r'[\[\(]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\]\)]'
        regex_points = re.findall(point_pattern, text)
        for p in regex_points:
            try:
                x, y = float(p[0]), float(p[1])
                points.append((x, y))
            except ValueError:
                pass

    return points

def where2place_process_results(doc, result):
    if len(result) == 0:
        return {"acc": {"score": 0.0, "sub_task": doc.get("sub_task", "unknown")}}
    pred_text = result[0]
    points = parse_points(pred_text)
    points_array = np.array(points)
    
    sub_task = doc.get("sub_task", "unknown")
    
    # Mask is now directly a PIL Image object in the doc
    mask_img = doc.get("mask")

    if mask_img is None:
        eval_logger.warning(f"No mask found for sample. Subtask: {sub_task}")
        return {"acc": {"score": 0.0, "sub_task": sub_task}}

    try:
        # Assuming mask is grayscale or relevant channel is 0-255
        mask = np.array(mask_img) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0] # Take first channel if multi-channel
    except Exception as e:
        eval_logger.error(f"Failed to process mask: {e}")
        return {"acc": {"score": 0.0, "sub_task": sub_task}}
    
    if model_id and "Qwen3-VL" in model_id:
        new_points = []
        for point in points:
            x, y = point
            h, w = mask_img.height, mask_img.width
            # unnormalize by 1000
            new_x = x / 1000. * w
            new_y = y / 1000. * h
            new_points.append((new_x, new_y))
        points = new_points
        points_array = np.array(points)

    acc = 0.0

    if len(points) > 0:
        # Check if points are within image bounds
        in_range = (
            (points_array[:, 0] >= 0)
            & (points_array[:, 0] < mask.shape[1])
            & (points_array[:, 1] >= 0)
            & (points_array[:, 1] < mask.shape[0])
        )
        
        # Use integer coordinates for indexing
        pts_x = points_array[in_range, 0].astype(int)
        pts_y = points_array[in_range, 1].astype(int)
        
        if len(pts_x) > 0:
            # Get mask values at valid points
            # Note: mask is (H, W), points are (x, y) -> mask[y, x]
            valid_values = mask[pts_y, pts_x]
            # valid_values = valid_values[valid_values > 0]
            
            # Combine with zeros for out-of-range points
            # points_array.shape[0] is total points
            # in_range.sum() is valid points
            # So we append (total - valid) zeros
            num_out_of_range = points_array.shape[0] - in_range.sum()
            all_values = np.concatenate([valid_values, np.zeros(num_out_of_range)])
            if len(all_values) < 1:
                acc = 0.0
            else:
                acc = float(all_values.mean())
            
        else:
            # All points out of range
            acc = 0.0
            
    # Call draw_result for visualization
    # We pass points as list of tuples for the function
    if len(points) > 0:
         draw_result(doc, mask_img, acc, points)
    else:
         # Even with no points, we might want to visualize the mask and 0 score
         draw_result(doc, mask_img, acc, [])

    return {
        "acc": {
            "score": acc,
            "sub_task": sub_task
        }
    }

def where2place_aggregate_results(results):
    """
    Aggregate results with support for sub-task breakdown.
    results: list of dicts, e.g. [{'score': 0.8, 'sub_task': 'living_room'}, ...]
    """
    total_score = 0.0
    count = 0
    subtask_stats = {}
    
    for res in results:
        # Handle case where res might be just the score (float) if something changed upstream, 
        # but based on process_results it should be a dict.
        if isinstance(res, dict):
            score = res.get("score", 0.0)
            sub_task = res.get("sub_task", "unknown")
        else:
            score = float(res)
            sub_task = "unknown"
            
        total_score += score
        count += 1
        
        if sub_task not in subtask_stats:
            subtask_stats[sub_task] = {"score": 0.0, "count": 0}
        subtask_stats[sub_task]["score"] += score
        subtask_stats[sub_task]["count"] += 1
        
    avg_score = total_score / count if count > 0 else 0.0
    
    eval_logger.info(f"Where2Place Evaluation Results:")
    eval_logger.info(f"Total Samples: {count}")
    eval_logger.info(f"Overall Average Score: {avg_score:.4f}")
    
    eval_logger.info("Subtask Breakdown:")
    for task, stats in sorted(subtask_stats.items()):
        task_avg = stats["score"] / stats["count"] if stats["count"] > 0 else 0.0
        eval_logger.info(f"  - {task}: {task_avg:.4f} (n={stats['count']})")
        
    return avg_score
