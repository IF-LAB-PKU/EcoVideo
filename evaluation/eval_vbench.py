import os
import json
import argparse
from pathlib import Path


VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
]

CUSTOM_INPUT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def parse_args():
    parser = argparse.ArgumentParser(description="EcoVideo evaluation with VBench")

    parser.add_argument(
        "--videos_path", type=str, required=True,
        help="Path to folder containing generated videos",
    )
    parser.add_argument(
        "--prompt_file", type=str, default=None,
        help="Text file with one prompt per line, or JSON dict {video_filename: prompt}",
    )
    parser.add_argument(
        "--dimension_list", type=str, nargs="+", default=None,
        help=f"VBench dimensions to evaluate (default: all custom-input supported). "
             f"Choices: {VBENCH_DIMENSIONS}",
    )
    parser.add_argument(
        "--output_dir", type=str, default="evaluation_results",
        help="Directory to save evaluation results (default: evaluation_results)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for evaluation (default: cuda)",
    )
    parser.add_argument(
        "--name", type=str, default="ecovideo",
        help="Name tag for this evaluation run (default: ecovideo)",
    )

    return parser.parse_args()


def load_prompts(prompt_file):
    if prompt_file is None:
        return []

    path = Path(prompt_file)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported JSON format in {prompt_file}")

    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def build_prompt_dict(videos_path, prompts):
    if isinstance(prompts, dict):
        return prompts

    video_files = sorted([
        f for f in os.listdir(videos_path)
        if Path(f).suffix.lower() in [".mp4", ".gif"]
    ])

    if len(prompts) == 0:
        return {}

    if len(prompts) < len(video_files):
        raise ValueError(
            f"Not enough prompts: {len(prompts)} prompts for {len(video_files)} videos. "
            f"VBench requires a prompt for every video in the folder. "
            f"Either add more prompts or remove extra videos."
        )

    prompt_dict = {}
    for i, video_file in enumerate(video_files):
        if i >= len(prompts):
            break
        prompt_dict[video_file] = prompts[i]

    return prompt_dict


def run_vbench(videos_path, prompt_dict, dimension_list, output_dir, device, name):
    try:
        import vbench as vbench_pkg
        from vbench import VBench
    except ImportError:
        raise ImportError(
            "vbench is not installed. Install it with:\n"
            "  pip install vbench\n"
            "  pip install detectron2@git+https://github.com/facebookresearch/detectron2.git"
        )

    os.makedirs(output_dir, exist_ok=True)

    if dimension_list is None:
        if len(prompt_dict) > 0:
            dimension_list = CUSTOM_INPUT_DIMENSIONS
        else:
            dimension_list = VBENCH_DIMENSIONS

    vbench_json = os.path.join(
        os.path.dirname(vbench_pkg.__file__), "VBench_full_info.json"
    )
    if not os.path.exists(vbench_json):
        raise FileNotFoundError(
            f"VBench_full_info.json not found at {vbench_json}. "
            "Please check your vbench installation: pip install vbench"
        )

    mode = "custom_input" if len(prompt_dict) > 0 else "vbench_standard"

    print(f"\n{'=' * 60}")
    print(f"VBench Evaluation")
    print(f"  Videos:     {videos_path}")
    print(f"  Mode:       {mode}")
    print(f"  Dimensions: {dimension_list}")
    print(f"  Output:     {output_dir}")
    print(f"{'=' * 60}\n")

    my_vbench = VBench(device, vbench_json, output_dir)
    my_vbench.evaluate(
        videos_path=videos_path,
        name=name,
        prompt_list=prompt_dict,
        dimension_list=dimension_list,
        mode=mode,
    )

    results_file = os.path.join(output_dir, f"{name}_eval_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
        print_results_table(results)
        return results

    return None


def print_results_table(results):
    print(f"\n{'=' * 60}")
    print(f"{'Dimension':<30} {'Score':>10}")
    print(f"{'-' * 40}")
    scores = []
    for dimension, data in results.items():
        if isinstance(data, list) and len(data) > 0:
            score = data[0] if isinstance(data[0], (int, float)) else "N/A"
        elif isinstance(data, (int, float)):
            score = data
        else:
            score = "N/A"
        if isinstance(score, float):
            print(f"  {dimension:<28} {score:>10.4f}")
            scores.append(score)
        else:
            print(f"  {dimension:<28} {str(score):>10}")
    if scores:
        avg = sum(scores) / len(scores)
        print(f"{'-' * 40}")
        print(f"  {'Average':<28} {avg:>10.4f}")
    print(f"{'=' * 60}\n")


def main():
    args = parse_args()

    prompts = load_prompts(args.prompt_file)
    prompt_dict = build_prompt_dict(args.videos_path, prompts)

    run_vbench(
        videos_path=args.videos_path,
        prompt_dict=prompt_dict,
        dimension_list=args.dimension_list,
        output_dir=args.output_dir,
        device=args.device,
        name=args.name,
    )


if __name__ == "__main__":
    main()
