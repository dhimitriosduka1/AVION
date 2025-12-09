# Part of the code is from https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/utils.py
# Modified by Yue Zhao

import os
import numpy as np

from avion.utils.evaluation_common import validate_zeroshot_cls, accuracy
from avion.utils.misc import generate_label_map
from avion.data.clip_dataset import VideoClassyDataset


def get_marginal_indexes(actions, mode):
    """For each verb/noun retrieve the list of actions containing that verb/name
    Input:
        mode: "verb" or "noun"
    Output:
        a list of numpy array of indexes. If verb/noun 3 is contained in actions 2,8,19,
        then output[3] will be np.array([2,8,19])
    """
    vi = []
    for v in range(actions[mode].max() + 1):
        vals = actions[actions[mode] == v].index.values
        if len(vals) > 0:
            vi.append(vals)
        else:
            vi.append(np.array([0]))
    return vi


def marginalize(probs, indexes):
    mprobs = []
    for ilist in indexes:
        mprobs.append(probs[:, ilist].sum(1))
    return np.array(mprobs).T


def get_mean_accuracy(cm):
    list_acc = []
    for i in range(len(cm)):
        acc = 0
        if cm[i, :].sum() > 0:
            acc = cm[i, i] / cm[i, :].sum()
        list_acc.append(acc)

    return 100 * np.mean(list_acc), 100 * np.trace(cm) / np.sum(cm)


def get_val_dataset(
    transform,
    video_chunk_length,
    clip_length,
    clip_stride,
    fused_decode_crop,
    crop_size,
    num_clips,
    threads,
):
    labels, mapping_vn2act = generate_label_map("ek100_cls")

    return (
        VideoClassyDataset(
            dataset="ek100_cls",
            root=f"{os.environ.get('EK100_META_DIR')}/video_320p_15sec",
            transform=transform,
            metadata=os.environ.get("VAL_METADATA"),
            is_training=False,
            label_mapping=mapping_vn2act,
            num_clips=num_clips,
            chunk_len=video_chunk_length,
            clip_length=clip_length,
            clip_stride=clip_stride,
            threads=threads,
            fast_rcc=fused_decode_crop,
            rcc_params=(crop_size,),
            is_trimmed=True,
        ),
        labels,
    )


def validate_zeroshot(
    val_loader,
    use_template,
    labels,
    model,
    tokenizer,
    disable_amp,
    fused_decode_crop,
    transform_gpu,
):
    print("=> starting ek100cls zeroshot evaluation")
    preds, targets = validate_zeroshot_cls(
        val_loader=val_loader,
        use_template=use_template,
        labels=labels,
        model=model,
        tokenizer=tokenizer,
        disable_amp=disable_amp,
        fused_decode_crop=fused_decode_crop,
        transform_gpu=transform_gpu,
    )
    print("=> finished ek100cls zeroshot evaluation")

    top1, top3, top5, top10 = accuracy(preds, targets, topk=(1, 3, 5, 10))
    print(
        f"=> ek100cls zeroshot evaluation results: top1={top1.item():.2f}, top3={top3.item():.2f}, top5={top5.item():.2f}, top10={top10.item():.2f}"
    )
    return {
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "top10": top10,
    }
