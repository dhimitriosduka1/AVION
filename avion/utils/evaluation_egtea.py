import os
from avion.utils.misc import generate_label_map
from avion.data.clip_dataset import VideoClassyDataset
from sklearn.metrics import confusion_matrix
from avion.utils.evaluation_common import validate_zeroshot_cls, get_mean_accuracy


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
    labels, mapping_vn2act = generate_label_map("egtea")

    return (
        VideoClassyDataset(
            dataset="egtea",
            root=f"{os.environ.get('EGTEA_DATA_DIR')}",
            transform=transform,
            metadata=os.environ.get("EGTEA_META_DIR"),
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
    print("=> starting egtea zeroshot evaluation")
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
    print("=> finished egtea zeroshot evaluation")

    preds, targets = preds.numpy(), targets.numpy()

    cm = confusion_matrix(targets, preds.argmax(axis=1))
    mean_class_acc, acc = get_mean_accuracy(cm)
    print(
        f"=> egtea zeroshot evaluation results: Mean Acc={mean_class_acc:.2f}, Top-1 Acc={acc:.2f}"
    )

    return {
        "mean_class_acc": mean_class_acc,
        "top1_acc": acc,
    }
