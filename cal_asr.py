import torch
import os
import csv
from collections import defaultdict


def cal_asr_all(pt_file_path):
    results_dict = torch.load(pt_file_path)

    clean_result = {}
    adv_result = {}

    for _, result in results_dict.items():
        for filename, v in result.get("clean", {}).items():
            clean_result[filename] = clean_result.get(filename, False) or v
        for filename, v in result.get("adv", {}).items():
            adv_result[filename] = adv_result.get(filename, False) or v

    clean_true_count = sum(v for v in clean_result.values())
    adv_true_count = sum(v for v in adv_result.values())

    clean_count = 1901 if "car" in pt_file_path else 1920

    asr = (clean_count - adv_true_count) / clean_count
    asr_normal = (clean_count - clean_true_count) / clean_count

    return clean_true_count, adv_true_count, asr_normal, asr


if __name__ == "__main__":

    # ================= Configuration =================
    # methods = [
    #     "results_armored_brown",
    #     "results_armored_green",
    #     "results_car_brown",
    #     "results_car_green",
    #     "results_das_armored",
    #     "results_das_car",
    #     "results_fca_armored",
    #     "results_fca_car"
    # ]

    # methods = [
    #     "results_car_green_aug"
    # ]

    # methods = ["results_car_green_wordnet"]

    methods = ["results_das_armored"]
    models = ["glip", "dino", "yolo"]
    thrs = [0.1, 0.2, 0.3, 0.4, 0.5]

    output_csv_raw = "/root/ovd-attack/asr_by_threshold_12prompts.csv"
    output_csv_mean = "/root/ovd-attack/asr_mean_12prompts.csv"
    # =================================================

    asr_pool = defaultdict(list)

    # ========== File 1: per-threshold results ==========
    with open(output_csv_raw, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "method", "model", "threshold",
            "clean_det", "adv_det",
            "normal_asr(%)", "asr(%)"
        ])

        for method in methods:
            for model in models:
                for thr in thrs:
                    if "das" in method or "fca" in method:
                        pt_path = (
                            f"/root/autodl-tmp/res/tgrs2/"
                            f"{method}/adv/{model}/train_single/"
                            f"{model}_multi_{thr}.pt"
                        )
                    else:
                        pt_path = (
                            f"/root/autodl-tmp/res/tgrs2/"
                            f"{method}/adv/{model}/train_multi/"
                            f"{model}_multi_{thr}.pt"
                        )

                    if not os.path.exists(pt_path):
                        continue

                    clean_det, adv_det, normal_asr, asr = cal_asr_all(pt_path)

                    writer.writerow([
                        method,
                        model,
                        thr,
                        clean_det,
                        adv_det,
                        round(normal_asr * 100, 2),
                        round(asr * 100, 2)
                    ])

                    asr_pool[(method, model)].append(asr * 100)

    # ========== File 2: mean ASR ==========
    with open(output_csv_mean, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "method", "model", "mean_asr(%)"
        ])

        for (method, model), asr_list in sorted(asr_pool.items()):
            mean_asr = sum(asr_list) / len(asr_list)
            writer.writerow([
                method,
                model,
                round(mean_asr, 2)
            ])

    print("Saved:")
    print(f" - Per-threshold results: {output_csv_raw}")
    print(f" - Mean ASR results: {output_csv_mean}")
