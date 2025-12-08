# Part of the code is from https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/utils.py
# Modified by Yue Zhao

import time
import scipy
import torch
import pickle
import numpy as np
import pandas as pd
import torch.cuda.amp as amp
from timm.utils import accuracy

from collections import OrderedDict
from avion.utils.meters import AverageMeter, ProgressMeter
from sklearn.metrics import confusion_matrix, top_k_accuracy_score


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


# Helper method to perform evaluation
def validate_cls(val_loader, transform_gpu, model, args, num_videos, val_dataset_name):
    batch_time = AverageMeter("Time", ":6.2f")
    data_time = AverageMeter("Data", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    metric_names = ["Acc@1", "Acc@5"]
    metrics = OrderedDict(
        [(name, AverageMeter(name, ":6.2f")) for name in metric_names]
    )
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Test: ",
    )

    model.eval()

    all_logits = [[] for _ in range(args.world_size)]
    all_probs = [[] for _ in range(args.world_size)]
    all_targets = [[] for _ in range(args.world_size)]
    total_num = 0

    with amp.autocast(enabled=not args.disable_amp):
        with torch.no_grad():
            end = time.time()
            for i, (videos, targets) in enumerate(val_loader):
                data_time.update(time.time() - end)

                if isinstance(videos, torch.Tensor):
                    videos = [
                        videos,
                    ]

                logits_allcrops = []
                for crop in videos:
                    crop = crop.cuda(args.gpu, non_blocking=True)

                    if args.fused_decode_crop and len(transform_gpu) > 0:
                        crop = crop.permute(0, 4, 1, 2, 3)
                        crop = transform_gpu(crop)

                    logits = model.module.encode_image(crop)
                    logits_allcrops.append(logits)

                logits_allcrops = torch.stack(logits_allcrops, 1)
                probs_allcrops = torch.softmax(logits_allcrops, dim=2)
                targets = targets.cuda(args.gpu, non_blocking=True)
                targets_repeated = torch.repeat_interleave(targets, len(videos))

                acc1, acc5 = accuracy(
                    torch.flatten(logits_allcrops, 0, 1), targets_repeated, topk=(1, 5)
                )
                metrics["Acc@1"].update(acc1.item(), targets_repeated.size(0))
                metrics["Acc@5"].update(acc5.item(), targets_repeated.size(0))

                gathered_logits = [
                    torch.zeros_like(logits_allcrops) for _ in range(args.world_size)
                ]
                gathered_probs = [
                    torch.zeros_like(probs_allcrops) for _ in range(args.world_size)
                ]
                gathered_targets = [
                    torch.zeros_like(targets) for _ in range(args.world_size)
                ]
                torch.distributed.all_gather(gathered_logits, logits_allcrops)
                torch.distributed.all_gather(gathered_probs, probs_allcrops)
                torch.distributed.all_gather(gathered_targets, targets)

                for j in range(args.world_size):
                    all_logits[j].append(gathered_logits[j].detach().cpu())
                    all_probs[j].append(gathered_probs[j].detach().cpu())
                    all_targets[j].append(gathered_targets[j].detach().cpu())
                total_num += logits_allcrops.shape[0] * args.world_size

                batch_time.update(time.time() - end)
                end = time.time()

                mem.update(torch.cuda.max_memory_allocated() // 1e9)

                if i % args.print_freq == 0:
                    progress.display(i)

    progress.synchronize()

    for j in range(args.world_size):
        all_logits[j] = torch.cat(all_logits[j], dim=0).numpy()
        all_probs[j] = torch.cat(all_probs[j], dim=0).numpy()
        all_targets[j] = torch.cat(all_targets[j], dim=0).numpy()

    all_logits_reorg, all_probs_reorg, all_targets_reorg = [], [], []

    for i in range(total_num):
        all_logits_reorg.append(all_logits[i % args.world_size][i // args.world_size])
        all_probs_reorg.append(all_probs[i % args.world_size][i // args.world_size])
        all_targets_reorg.append(all_targets[i % args.world_size][i // args.world_size])

    all_logits = np.stack(all_logits_reorg, axis=0)
    all_probs = np.stack(all_probs_reorg, axis=0)
    all_targets = np.stack(all_targets_reorg, axis=0)
    all_logits = all_logits[:num_videos, :].mean(axis=1)
    all_probs = all_probs[:num_videos, :].mean(axis=1)
    all_targets = all_targets[:num_videos,]

    # if args.pickle_filename != "":
    #     prob_dict = {"logits": all_logits, "probs": all_probs, "targets": all_targets}
    #     pickle.dump(prob_dict, open(args.pickle_filename, "wb"))

    for s, all_preds in zip(["logits", " probs"], [all_logits, all_probs]):
        if s == "logits":
            all_preds = scipy.special.softmax(all_preds, axis=1)

        acc1 = top_k_accuracy_score(
            all_targets, all_preds, k=1, labels=np.arange(0, args.num_classes)
        )
        acc5 = top_k_accuracy_score(
            all_targets, all_preds, k=5, labels=np.arange(0, args.num_classes)
        )

        print(
            "[Average {s}] {dataset} * Acc@1 {top1:.3f} Acc@5 {top5:.3f}".format(
                s=s, dataset=val_dataset_name, top1=acc1, top5=acc5
            )
        )

        cm = confusion_matrix(all_targets, all_preds.argmax(axis=1))
        mean_acc, acc = get_mean_accuracy(cm)
        print("Mean Acc. = {:.3f}, Top-1 Acc. = {:.3f}".format(mean_acc, acc))

        noun_acc = 0.0
        verb_acc = 0.0
        if val_dataset_name == "ek100_cls":
            vi = get_marginal_indexes(args.actions, "verb")
            ni = get_marginal_indexes(args.actions, "noun")
            verb_scores = marginalize(all_preds, vi)
            noun_scores = marginalize(all_preds, ni)

            target_to_verb = np.array(
                [args.mapping_act2v[a] for a in all_targets.tolist()]
            )
            target_to_noun = np.array(
                [args.mapping_act2n[a] for a in all_targets.tolist()]
            )

            cm = confusion_matrix(target_to_verb, verb_scores.argmax(axis=1))
            _, verb_acc = get_mean_accuracy(cm)
            print("Verb Acc@1: {:.3f}".format(acc))

            cm = confusion_matrix(target_to_noun, noun_scores.argmax(axis=1))
            _, noun_acc = get_mean_accuracy(cm)
            print("Noun Acc@1: {:.3f}".format(acc))

    return {
        "acc1": metrics["Acc@1"].avg,
        "acc5": metrics["Acc@5"].avg,
        "mean_acc": mean_acc,
        "mean_acc_noun": noun_acc,
        "mean_acc_verb": verb_acc,
    }
