import os
import torch
from torch.cuda import amp
import numpy as np
import avion.utils.distributed as dist_utils
from tqdm import tqdm
from avion.data.clip_dataset import VideoCaptionDatasetMCQ
from sklearn.metrics import confusion_matrix
from avion.utils.evaluation_common import validate_zeroshot_cls, get_mean_accuracy


def egomcq_accuracy_metrics(preds, labels, types):
    metrics = {}
    type_list = torch.unique(types)
    group_list = ["Intra-video", "Inter-video"]
    for type_i, group_i in zip(type_list, group_list):
        correct = 0
        total = 0
        for pred, label, type in zip(preds, labels, types):
            if type == type_i:
                pred_ = torch.argmax(pred)
                if pred_.item() == label.item():
                    correct += 1
                total += 1
        accuracy = correct / total
        metrics[group_i] = accuracy * 100
    return metrics


def validate_mcq(val_loader, model, fused_decode_crop, transform_gpu, disable_amp):
    model.eval()

    with amp.autocast(enabled=not disable_amp):
        with torch.no_grad():
            print("=> start forwarding")
            all_preds = []
            all_gts = []
            all_types = []

            for i, inputs in tqdm(enumerate(val_loader), desc="Validation MCQ"):
                texts_query = inputs[0].cuda(non_blocking=True)
                frames_options = inputs[1].cuda(non_blocking=True)

                answer = inputs[3]
                q_type = inputs[4]
                if len(inputs) == 7:
                    masks_query = inputs[5].cuda(non_blocking=True)
                else:
                    masks_query = None

                batch_size = frames_options.shape[0]
                frames_options = frames_options.view(-1, *frames_options.shape[2:])

                if fused_decode_crop and len(transform_gpu) > 0:
                    frames_options = frames_options.permute(0, 4, 1, 2, 3)
                    frames_options = transform_gpu(frames_options)

                image_features = dist_utils.get_model(model).encode_image(
                    frames_options
                )

                image_features = image_features.view(
                    batch_size, -1, *image_features.shape[1:]
                )

                if masks_query is not None:
                    query_features = dist_utils.get_model(model).encode_text(
                        texts_query, attention_mask=masks_query
                    )
                else:
                    query_features = dist_utils.get_model(model).encode_text(
                        texts_query
                    )

                all_gts.append(answer)
                all_types.append(q_type)
                for j in range(batch_size):
                    similarity_matrix = torch.matmul(
                        query_features[j], image_features[j].T
                    )
                    similarity_matrix = similarity_matrix.cpu().detach()
                    all_preds.append(similarity_matrix)

            all_preds = torch.stack(all_preds)
            all_gts = torch.cat(all_gts)
            all_types = torch.cat(all_types)
            metrics = egomcq_accuracy_metrics(all_preds, all_gts, all_types)

    return metrics


def get_val_dataset(
    transform,
    clip_length,
    clip_stride,
    tokenizer,
    chunk_length,
    threads,
    fused_decode_crop,
    crop_size,
    num_clips,
):
    assert clip_length == 4.0
    return VideoCaptionDatasetMCQ(
        "ego4d_mcq",
        root=os.environ.get("EGO4D_MCQ_DATA_DIR"),
        metadata=os.environ.get("EGO4D_MCQ_META_DIR"),
        transform=transform,
        is_training=False,
        tokenizer=tokenizer,
        chunk_len=chunk_length,
        clip_length=clip_length,
        clip_stride=clip_stride,
        threads=threads,
        fast_rcc=fused_decode_crop,
        rcc_params=(crop_size,),
        num_clips=num_clips,
    )


def validate_zeroshot(
    val_loader,
    model,
    fused_decode_crop,
    transform_gpu,
    disable_amp,
):
    print("=> starting egomcq zeroshot evaluation")
    results = validate_mcq(
        val_loader=val_loader,
        model=model,
        fused_decode_crop=fused_decode_crop,
        transform_gpu=transform_gpu,
        disable_amp=disable_amp
    )
    print("=> finished egomcq zeroshot evaluation")
    print(f"=> egomcq zeroshot evaluation results: {results}")

    return results
