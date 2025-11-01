import torch
import numpy as np
import torch.cuda.amp as amp
import avion.utils.distributed as dist_utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


def validate_zeroshot(
    val_loader, templates, labels, model, tokenizer, disable_amp=False
):
    model.eval()

    all_outputs = []
    all_targets = []
    all_vis_features = []
    print("=> encoding captions")
    with amp.autocast(enabled=not disable_amp):
        with torch.no_grad():
            text_features = []
            for label in tqdm(labels, desc="Encoding captions"):
                if isinstance(label, list):
                    texts = [tmpl.format(lbl) for tmpl in templates for lbl in label]
                else:
                    texts = [tmpl.format(label) for tmpl in templates]
                texts = tokenizer(texts)

                texts = texts.cuda(non_blocking=True)
                texts = texts.view(-1, 77).contiguous()

                class_embeddings = dist_utils.get_model(model).encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )

                text_features.append(class_embeddings)

            text_features = torch.stack(text_features, dim=0)

            print("=> start forwarding")
            for _, (images, target) in tqdm(enumerate(val_loader), desc="Forwarding"):
                if isinstance(images, torch.Tensor):
                    images = images.cuda(non_blocking=True)

                    target = target.cuda(non_blocking=True)

                    image_features = dist_utils.get_model(model).encode_image(images)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    all_vis_features.append(image_features)

                    logits_per_image = image_features @ text_features.t()

                else:
                    target = target.cuda(non_blocking=True)
                    images_list = images
                    logits_all_clips = []
                    for images in images_list:
                        images = images.cuda(non_blocking=True)

                        image_features = dist_utils.get_model(model).encode_image(
                            images
                        )
                        image_features = image_features / image_features.norm(
                            dim=-1, keepdim=True
                        )

                        logits_per_image = image_features @ text_features.t()
                        logits_all_clips.append(logits_per_image)

                    logits_all_clips = torch.stack(logits_all_clips, dim=0)
                    logits_per_image = logits_all_clips.max(0).values
                    logits_per_image = torch.softmax(logits_per_image, dim=1)

                all_outputs.append(logits_per_image.cpu())
                all_targets.append(target.cpu())

    return torch.cat(all_outputs), torch.cat(all_targets)


def get_mean_accuracy(cm):
    list_acc = []
    for i in range(len(cm)):
        acc = 0
        if cm[i, :].sum() > 0:
            acc = cm[i, i] / cm[i, :].sum()
        list_acc.append(acc)

    return 100 * np.mean(list_acc), 100 * np.trace(cm) / np.sum(cm)


def validate_mcq(val_loader, model, use_half=False):
    model.eval()

    if use_half:
        model.half()

    with torch.no_grad():
        all_preds = []
        all_gts = []
        all_types = []

        for i, inputs in enumerate(val_loader):
            texts_query = inputs[0].cuda(non_blocking=True)
            frames_options = inputs[1].cuda(non_blocking=True)
            if use_half:
                frames_options = frames_options.half()
            answer = inputs[3]
            q_type = inputs[4]
            if len(inputs) == 7:
                masks_query = inputs[5].cuda(non_blocking=True)
            else:
                masks_query = None

            batch_size = frames_options.shape[0]

            frames_options = frames_options.view(-1, *frames_options.shape[2:])
            image_features = dist_utils.get_model(model).encode_image(frames_options)
            image_features = image_features.view(
                batch_size, -1, *image_features.shape[1:]
            )

            if masks_query is not None:
                query_features = dist_utils.get_model(model).encode_text(
                    texts_query, attention_mask=masks_query
                )
            else:
                query_features = dist_utils.get_model(model).encode_text(texts_query)

            all_gts.append(answer)
            all_types.append(q_type)

            for j in range(batch_size):
                similarity_matrix = torch.matmul(query_features[j], image_features[j].T)
                similarity_matrix = similarity_matrix.cpu().detach()
                all_preds.append(similarity_matrix)

        all_preds = torch.stack(all_preds)
        all_gts = torch.cat(all_gts)
        all_types = torch.cat(all_types)
        metrics = egomcq_accuracy_metrics(all_preds, all_gts, all_types)

        return metrics


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


def map(submission_array, gt_array):
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
    return map(fix, gt_array)


def validate_charades_ego(val_loader, labels, model, tokenizer):
    charades_ego_preds, charades_ego_targets = validate_zeroshot(
        val_loader,
        ["{}"],
        labels,
        model,
        tokenizer,
    )

    charades_ego_mAP, _, _ = charades_map(
        charades_ego_preds.numpy(), charades_ego_targets.numpy()
    )

    print("Charades Ego mAP: {:.3f}".format(charades_ego_mAP))

    return charades_ego_mAP


def validate_egtea(val_loader, labels, model, tokenizer):
    egtea_preds, egtea_targets = validate_zeroshot(
        val_loader, ["{}"], labels, model, tokenizer
    )

    cm = confusion_matrix(egtea_targets.numpy(), egtea_preds.numpy().argmax(axis=1))

    egtea_mean_class_acc, egtea_top1_acc = get_mean_accuracy(cm)

    print(
        "EGTEA Mean Acc: {:.3f}, Top-1 Acc: {:.3f}".format(
            egtea_mean_class_acc, egtea_top1_acc
        )
    )

    return egtea_mean_class_acc, egtea_top1_acc, cm


def plot_confusion_matrix(cm, labels):
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,  # Add labels for predicted classes
        yticklabels=labels,  # Add labels for true classes
    )
    plt.title("EGTEA Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")  # Optional: rotate labels for readability
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig
