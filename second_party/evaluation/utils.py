import torch
import numpy as np
import torch.amp as amp
import avion.utils.distributed as dist_utils


def validate_zeroshot(
    val_loader, templates, labels, model, tokenizer, use_half=False, disable_amp=False
):
    model.eval()

    if use_half:
        model = model.half()

    all_outputs = []
    all_targets = []
    all_vis_features = []
    print("=> encoding captions")
    with amp.autocast(enabled=not disable_amp):
        with torch.no_grad():
            text_features = []
            for label in labels:
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
            for _, (images, target) in enumerate(val_loader):
                if isinstance(images, torch.Tensor):
                    images = images.cuda(non_blocking=True)

                    if use_half:
                        images = images.half()

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
                        if use_half:
                            images = images.half()

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
