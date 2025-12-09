import torch
from torch.cuda import amp
from tqdm import tqdm
import avion.utils.distributed as dist_utils


def get_templates(use_template=True):
    return ["#C C {}", "#C {}"] if use_template else ["{}"]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate_zeroshot_cls(
    val_loader,
    use_template,
    labels,
    model,
    tokenizer,
    fused_decode_crop,
    transform_gpu,
    disable_amp=False,
):
    model.eval()

    templates = get_templates(use_template)

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

                    # compute output
                    if fused_decode_crop and len(transform_gpu) > 0:
                        images = images.permute(0, 4, 1, 2, 3)
                        images = transform_gpu(images)

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

                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    world_size = torch.distributed.get_world_size()
                    gathered_logits = [
                        torch.zeros_like(logits_per_image) for _ in range(world_size)
                    ]
                    gathered_targets = [
                        torch.zeros_like(target) for _ in range(world_size)
                    ]
                    torch.distributed.all_gather(gathered_logits, logits_per_image)
                    torch.distributed.all_gather(gathered_targets, target)
                    for r in range(world_size):
                        all_outputs.append(gathered_logits[r].detach().cpu())
                        all_targets.append(gathered_targets[r].detach().cpu())
                else:
                    all_outputs.append(logits_per_image.detach().cpu())
                    all_targets.append(target.detach().cpu())

    all_outputs = torch.cat(all_outputs) if len(all_outputs) > 0 else None
    all_targets = torch.cat(all_targets) if len(all_targets) > 0 else None

    return all_outputs, all_targets
