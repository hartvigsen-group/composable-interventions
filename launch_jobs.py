import os
import ast
import json
import random
import numpy as np
import pandas as pd

import wandb
from tqdm import tqdm
from datetime import datetime
from itertools import combinations


def get_all_combinations(lst):
    all_combs = []
    for r in range(1, len(lst) + 1):
        all_combs.extend(combinations(lst, r))

    return all_combs


def set_tag(experiment_row):
    if not isinstance(experiment_row, dict):
        experiment_row = experiment_row.to_dict()

    if experiment_row["interventions"] in [None, np.nan]:
        return "NONE"

    intervention_categories = None
    if isinstance(experiment_row["interventions"], str):
        intervention_categories = ast.literal_eval(experiment_row["interventions"])
    else:
        intervention_categories = experiment_row["interventions"]

    interventions = []
    for category in intervention_categories:
        intervention = (
            experiment_row.get(category, category).upper() if category != "compress" else experiment_row.get("compression", category).upper()
        )
        if intervention in ["AWQ", "GPTQ"]:
            intervention += str(int(experiment_row["wbits"])) + "bit"
        if intervention in ["WANDA", "SPARSEGPT"]:
            intervention += str(int(experiment_row["sparsity_ratio"] * 100)) + "%"

        interventions.append(intervention)

    if len(interventions) == 0:
        interventions.append("NONE")

    return "_".join(interventions)


def get_relevant_cols():
    setting_columns = [
        "tag",
        "edit",
        "compression",
        "unlearn",
        "_timestamp",
        "interventions",
        "edit_set",
        "edit_dataset",
        "rmu_layer_id",
        "wbits",
        "sparsity_ratio",
        "model_name",
    ]
    evaluation_columns = [
        "qa_question_count_limit",
        "mmlu accuracy",
        "wmdp_bio accuracy",
        "wmdp_cyber accuracy",
        "PPL",
        "PPL edits",
        "PPl QA",
        "Generalization",
        "FLOPs",
        "Success recall",
        "Generalization recall",
        "Locality",
        "Average bits",
        "Rewrite accuracy",
        "PPl edits unmasked",
        "Local recall",
        "Latency",
    ]
    return setting_columns + evaluation_columns


def should_keep_frame(frame):
    if frame["edit_dataset"] == "zsre":
        return True

    if "edit" not in frame["interventions"]:
        return True

    print(f"Skipping {frame['tag']} for edit dataset {frame['edit_dataset']}")
    return False


if __name__ == "__main__":
    models = ["01-ai/Yi-1.5-9B-Chat"]

    # Interventions
    editing_interventions = ["memit"]
    unlearn_interventions = []
    pruning_interventions = ["wanda"]
    quantization_interventions = ["awq"]
    all_interventions = editing_interventions + unlearn_interventions + pruning_interventions + quantization_interventions
    print(all_interventions)

    # Intervention Settings
    pruning_levels = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    quant_levels = [2, 3, 4, 5, 6, 8]
    rmu_setting_overrides = {
        "rmu_alpha": "[1000,1000]",
        "rmu_layer_id": 6,
        "rmu_max_num_batches": 450,
    }

    all_intervention_combinations = get_all_combinations(all_interventions)

    max_num_interventions = 2
    run_configurations = []
    for intervention_combination in all_intervention_combinations:
        if len(intervention_combination) > max_num_interventions:
            continue

        count_interventions = len(intervention_combination)
        all_compression_interventions = pruning_interventions + quantization_interventions
        is_double_compression = sum([technique in all_compression_interventions for technique in intervention_combination]) == 2
        if is_double_compression:
            continue

        intervention_orderings = {intervention_combination, tuple(reversed(intervention_combination))}
        for model_name in models:
            for ordering in intervention_orderings:
                if contains_pruning := any([technique in pruning_interventions for technique in ordering]):
                    for sparsity_ratio in pruning_levels:
                        run_configurations.append(
                            {"interventions": ordering, "sparsity_ratio": sparsity_ratio, "wbits": 16, "model_name": model_name}
                        )
                elif contains_quantization := any([technique in quantization_interventions for technique in ordering]):
                    for wbits in quant_levels:
                        run_configurations.append({"interventions": ordering, "sparsity_ratio": 0, "wbits": wbits, "model_name": model_name})

                default_sparsity_ratio = 0
                default_wbits = 16

    # add one without interventions, just the model
    run_configurations.append({"interventions": [], "sparsity_ratio": 0, "wbits": 16, "model_name": model_name})

    skip_previous_runs = False
    previous_runs = None
    wandb_api = None if skip_previous_runs else wandb.Api()
    # TODO

    project_paths = ["dri-ice/Composable_Interventions"]
    filter_dict = {"state": "finished"}
    runs_frames = []
    for project_path in project_paths:
        runs = wandb_api.runs(project_path, filters=filter_dict)
        for run in tqdm(runs, desc=project_path):
            try:
                run_start_datetime = datetime.fromtimestamp(run.summary_metrics["_timestamp"])
                start_cutoff = datetime.strptime("2024-08-1 00:00:00", "%Y-%m-%d %H:%M:%S")
                end_cutoff = datetime.strptime("2024-10-19 00:00:00", "%Y-%m-%d %H:%M:%S")
                if run_start_datetime > end_cutoff:
                    continue
                if run_start_datetime < start_cutoff:
                    break

                skip_tags = ["test", "hparam_search"]
                should_skip = False
                for tag in skip_tags:
                    if tag in run.config["tag"].lower():
                        should_skip = True

                if should_skip:
                    continue

                config_frame = pd.DataFrame([run.config])
                summary_frame = pd.DataFrame([run.summary_metrics])
                combined_frame = pd.concat([config_frame, summary_frame], axis=1)
                runs_frames.append(combined_frame)
            except Exception:
                # print(f"Error processing run {run.id}: {e}")
                continue

    print(f"Total runs: {len(runs_frames)}")

    relevant_columns = get_relevant_cols()
    all_runs_df = pd.concat(runs_frames, ignore_index=True)[relevant_columns]
    all_runs_df["tag"] = all_runs_df.apply(set_tag, axis=1)
    print(all_runs_df.value_counts("tag"))
    all_runs_df

    filtered_run_configurations = []
    for run_config in run_configurations:
        run_model = run_config["model_name"]
        run_tag = set_tag(run_config)
        historical_model_runs = all_runs_df[all_runs_df["model_name"] == run_model]
        historical_model_runs = historical_model_runs[historical_model_runs["tag"] == run_tag]
        if len(historical_model_runs) == 0:
            filtered_run_configurations.append(run_config)
            continue

        print(f"Skipping {run_model} with tag {run_tag} as it already exists")

    before_count = len(run_configurations)
    print(f"Total runs to execute: {len(filtered_run_configurations)} after skipping {before_count - len(filtered_run_configurations)} previous runs")

    commands = []
    use_wandb = True
    is_slurm = True
    python_path = "~/miniconda3/envs/lm-compose/bin/python"

    run_commands = []
    for run_config in filtered_run_configurations:
        intervention_type_map = {"memit": "edit", "rmu": "unlearn", "wanda": "compression", "awq": "compression"}
        interventions_arg = [intervention_type_map[intervention] for intervention in run_config["interventions"]]
        interventions_arg_str = f"interventions={interventions_arg}".replace("'", "").replace(" ", "")
        category_args = [f"{intervention_type_map[intervention]}={intervention}" for intervention in run_config["interventions"]]
        category_args_str = " ".join(category_args)

        compress_args = []
        for intervention in run_config["interventions"]:
            if intervention in pruning_interventions:
                compress_args.append(f"sparsity_ratio={run_config['sparsity_ratio']}")
            elif intervention in quantization_interventions:
                compress_args.append(f"wbits={run_config['wbits']}")

        command_prefix = "sbatch run_exp.sh" if is_slurm else f"{python_path} -m lm_compose"
        command = command_prefix + f" model_name={run_config['model_name']} {interventions_arg_str} {category_args_str} {' '.join(compress_args)}"

        involves_rmu = any([intervention in run_config["interventions"] for intervention in unlearn_interventions])
        if involves_rmu:
            for key, value in rmu_setting_overrides.items():
                command += f" {key}={value}"

        tag = set_tag(run_config)
        command += f" tag={tag}"

        if use_wandb:
            command += " wandb=online"

        print(command)
        run_commands.append(command)

    print(f"Total number of runs: {len(run_configurations)}")
    print(json.dumps(run_configurations, indent=4))

    # Skip WANDA due to memory issues
    # run_commands = [command for command in run_commands if "compression=wanda" not in command]
    # print(f"Total number of runs: {len(run_commands)}")

    random.shuffle(run_commands)

    for command in tqdm(run_commands, desc="Running Experiments"):
        print(f"Executing: {command}")
        code = os.system(command)
        if code != 0:
            raise Exception(f"Command failed with code: {code}")
