import os
import numpy as np
from avion.utils.misc import generate_label_map
from avion.data.clip_dataset import VideoClassyDataset
from avion.utils.evaluation_common import validate_zeroshot_cls, accuracy


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
    labels, mapping_vn2act = generate_label_map("charades_ego")

    return (
        VideoClassyDataset(
            dataset="charades_ego",
            root=f"{os.environ.get('CHARADES_DATA_DIR')}",
            metadata=os.environ.get("CHARADES_META_DIR"),
            transform=transform,
            is_training=False,
            label_mapping=mapping_vn2act,
            num_clips=num_clips,
            chunk_len=video_chunk_length,
            clip_length=clip_length,
            clip_stride=clip_stride,
            threads=threads,
            fast_rcc=fused_decode_crop,
            rcc_params=(crop_size,),
            is_trimmed=False,
        ),
        labels,
    )


def compute_map(submission_array, gt_array):
    """Returns mAP, weighted mAP, and AP array"""
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float("nan"))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    w_ap = m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float)
    return m_ap, w_ap, m_aps


def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return compute_map(fix, gt_array)


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
    print("=> starting charades zeroshot evaluation")
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
    print("=> finished charades zeroshot evaluation")

    preds, targets = preds.numpy(), targets.numpy()
    m_ap, _, __ = charades_map(preds, targets)
    print(f"=> charades zeroshot evaluation results: mAP {m_ap:.4f}")

    return {"mAP": m_ap}
