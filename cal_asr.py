import torch

def cal_asr_all(pt_file_path):
    """
    {
        "car": {"clean": {...}, "adv": {...}},
        "automobile": {"clean": {...}, "adv": {...}},
        ...
    }
    """
    print('*' * 40)
    results_dict = torch.load(pt_file_path)
    clean_result = {}
    adv_result = {}

    for text, result in results_dict.items():
        clean_dict = result.get("clean", {})
        adv_dict = result.get("adv", {})

        for filename, v in clean_dict.items():
            clean_result[filename] = clean_result.get(filename, False) or v

        for filename, v in adv_dict.items():
            adv_result[filename] = adv_result.get(filename, False) or v

    clean_true_count = sum(1 for v in clean_result.values() if v)
    adv_true_count = sum(1 for v in adv_result.values() if v)

    clean_count = 1920
    asr = (clean_count - adv_true_count) / max(clean_count, 1)
    asr_normal = (clean_count - clean_true_count) / max(clean_count, 1)

    print(f"Clean detections: {clean_true_count}")
    print(f"Adv detections: {adv_true_count}")
    print(f"Normal ASR: {asr_normal:.2%}")
    print(f"ASR: {asr:.2%}")

    return clean_true_count, adv_true_count, asr

if __name__ == '__main__':
    for score in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _, _, res_multi = cal_asr_all(f'results_iou/adv/glip/train_single/glip_multi_{score}.pt')
    # print('*' * 40)
    # for score in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     _, _, res_single = cal_asr_all(f'/root/ovd-attack/results_clamp/adv/dino/train_single/dino_multi_{score}.pt')
