#!/usr/bin/env python3
"""
Download annotations from Label Studio project to JSONL format.

Usage:
    python -m human_eval.download_annotations --project-id 6 --output annotation_exports/human/project_6_annotations.jsonl
    python -m human_eval.download_annotations --project-id 9 --output annotation_exports/human/project_9_annotations.jsonl
    python -m human_eval.download_annotations --project-id 11 --output annotation_exports/human/project_11_annotations.jsonl # khmer
    python -m human_eval.download_annotations --project-id 13 --output annotation_exports/human/project_13_annotations.jsonl # ukrainian
    python -m human_eval.download_annotations --project-id 14 --output annotation_exports/human/project_14_annotations.jsonl # javanese
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
from pathlib import Path

from human_eval.label_studio_client import LabelStudioClient


def parse_annotation(annotation: dict) -> dict:
    """Parse a single annotation result into a flat structure."""
    result = {}
    
    for item in annotation.get("result", []):
        from_name = item.get("from_name", "")
        item_type = item.get("type", "")
        value = item.get("value", {})
        
        if item_type == "number":
            # Numeric scores (accuracy, fluency, adaptedness)
            result[from_name] = value.get("number")
        elif item_type == "textarea":
            # Text fields (comments)
            text_values = value.get("text", [])
            result[from_name] = text_values[0] if text_values else ""
        elif item_type == "labels":
            # Error span labels
            if "error_spans" not in result:
                result["error_spans"] = []
            result["error_spans"].append({
                "labels": value.get("labels", []),
                "start": value.get("start"),
                "end": value.get("end"),
                "text": value.get("text", ""),
            })
        
        if from_name == "is_critical_failure" and value:
            # by definition, a critical failure has 0 on accuracy / fluency / adaptedness
            result["is_critical_failure"] = True
    
    return result


def export_to_jsonl(tasks: list[dict], output_path: Path) -> int:
    """Export tasks with annotations to JSONL format.
    
    Args:
        tasks: List of tasks from Label Studio export
        output_path: Path to write JSONL file
    
    Returns:
        Number of annotated tasks exported
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    annotated_count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for task in tasks:
            annotations = task.get("annotations", [])
            
            # Skip tasks without annotations
            if not annotations:
                continue
            
            # Get the most recent completed annotation
            completed_annotations = [a for a in annotations if a.get("was_cancelled") is False]
            if not completed_annotations:
                continue
            
            # Use the most recent annotation
            annotation = max(completed_annotations, key=lambda a: a.get("id", 0))
            
            # Build export row
            row = {
                "task_id": task.get("id"),
                # Original task data
                "source_text": task.get("data", {}).get("source_text"),
                "user_instruction": task.get("data", {}).get("user_instruction"),
                "model_translation": task.get("data", {}).get("model_translation"),
                "reference_translation": task.get("data", {}).get("reference_translation"),
                # Metadata
                "meta": task.get("meta", {}),
                # Annotation info
                "annotation_id": annotation.get("id"),
                "annotator_id": annotation.get("completed_by"),
                "created_at": annotation.get("created_at"),
                "updated_at": annotation.get("updated_at"),
                # Parsed annotation results
                **parse_annotation(annotation),
            }
            # if row has is_critical_failure, set fields that end in `_score` scores to 0
            if row.get("is_critical_failure"):
                for key in row.keys():
                    if key.endswith("_score"):
                        row[key] = 0
            
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            annotated_count += 1
    
    return annotated_count


def main():
    parser = argparse.ArgumentParser(description="Export Label Studio annotations to JSONL")
    parser.add_argument("--project-id", type=int, default=6, help="Label Studio project ID")
    parser.add_argument("--output", type=str, help="Output JSONL file path")
    args = parser.parse_args()
    
    # Get credentials
    base_url = os.environ.get("LABEL_STUDIO_HOST", "http://localhost:8080")
    api_key = os.environ.get("LABEL_STUDIO_API_KEY")
    
    if not api_key:
        raise ValueError("Please set LABEL_STUDIO_API_KEY environment variable")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"annotation_exports/project_{args.project_id}_annotations.jsonl")
    
    print(f"🔗 Connecting to Label Studio at {base_url}...")
    client = LabelStudioClient(base_url, api_key)
    
    print(f"📥 Exporting tasks from project {args.project_id}...")
    tasks = client.export_tasks(args.project_id)
    print(f"   Retrieved {len(tasks)} total tasks")
    
    print(f"💾 Writing annotations to {output_path}...")
    annotated_count = export_to_jsonl(tasks, output_path)
    
    print(f"✅ Exported {annotated_count} annotated tasks to {output_path}")
    
    # Summary stats
    unannotated = len(tasks) - annotated_count
    if unannotated > 0:
        print(f"   ({unannotated} tasks without annotations were skipped)")


if __name__ == "__main__":
    main()
