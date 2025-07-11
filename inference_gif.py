import os
import imageio
import mmcv
from mmdet.registry import VISUALIZERS
import numpy as np
from mmdet.apis import init_detector, inference_detector
import cv2


def mmdet_infer_from_gif(config_file, checkpoint_file, input_gif, output_gif, device='cuda:0', score_thr=0.5):
    model = init_detector(config_file, checkpoint_file, device=device)
    gif_frames = imageio.mimread(input_gif)
    print(f"[INFO] Loaded {len(gif_frames)} frames from {input_gif}")

    output_frames = []
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    for idx, frame in enumerate(gif_frames):
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        result = inference_detector(model, frame_bgr, text_prompt='car')
        visualizer.add_datasample(
            name='results',
            image=frame_bgr,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=score_thr,
            out_file="results/result.jpg",)
        print(result)
        break
    #     vis_frame = model.show_result(
    #         frame_bgr, result, score_thr=score_thr, show=False)
    #     vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
    #     output_frames.append(vis_frame_rgb)
    #     print(f"[INFO] Processed frame {idx + 1}/{len(gif_frames)}")

    # imageio.mimsave(output_gif, output_frames, duration=0.1)
    # print(f"[INFO] Inference GIF saved to {output_gif}")


if __name__ == '__main__':
    config_file = 'mmdetection/configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py'
    checkpoint_file = 'weights/groundingdino/groundingdino_swint_ogc_mmdet-822d7e9d.pth'
    input_gif = 'results/result.gif'
    output_gif = 'results/result_det.gif'
    score_thr = 0.5
    mmdet_infer_from_gif(config_file, checkpoint_file,
                         input_gif, output_gif, score_thr=score_thr)
