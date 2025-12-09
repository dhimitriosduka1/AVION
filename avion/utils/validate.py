import avion.utils.evaluation_ek100cls as eval_ek100cls
import avion.utils.evaluation_egtea as eval_egtea
import avion.utils.evaluation_charades as eval_charades
import avion.utils.evaluation_egomcq as eval_egomcq
from avion.utils.evaluation_ek100mir import validate_mir


def validate_all(model, criterion, tokenizer, val_transform_gpu, args, val_loaders):
    results = {}

    if "ego4d_mir" in val_loaders:
        results["ego4d_mir"] = validate_mir(
            val_loaders["ego4d_mir"], val_transform_gpu, model, criterion, args
        )

    if "ego4d_cls" in val_loaders:
        loader, labels = val_loaders["ego4d_cls"]
        results["ego4d_cls"] = eval_ek100cls.validate_zeroshot(
            val_loader=loader,
            use_template=True,
            labels=labels,
            model=model,
            tokenizer=tokenizer,
            disable_amp=args.disable_amp,
            fused_decode_crop=args.fused_decode_crop,
            transform_gpu=val_transform_gpu,
        )

    if "egtea" in val_loaders:
        loader, labels = val_loaders["egtea"]
        results["egtea"] = eval_egtea.validate_zeroshot(
            val_loader=loader,
            use_template=True,
            labels=labels,
            model=model,
            tokenizer=tokenizer,
            disable_amp=args.disable_amp,
            fused_decode_crop=args.fused_decode_crop,
            transform_gpu=val_transform_gpu,
        )

    if "charades" in val_loaders:
        loader, labels = val_loaders["charades"]
        results["charades"] = eval_charades.validate_zeroshot(
            val_loader=loader,
            use_template=True,
            labels=labels,
            model=model,
            tokenizer=tokenizer,
            disable_amp=args.disable_amp,
            fused_decode_crop=args.fused_decode_crop,
            transform_gpu=val_transform_gpu,
        )

    if "egomcq" in val_loaders:
        results["egomcq"] = eval_egomcq.validate_zeroshot(
            val_loader=val_loaders["egomcq"],
            model=model,
            fused_decode_crop=args.fused_decode_crop,
            transform_gpu=val_transform_gpu,
            disable_amp=args.disable_amp,
        )

    # Print results
    for key, res in results.items():
        print(f"{key}_val_results: {res}")

    wandb_data = {}
    for key, res in results.items():
        wandb_data.update({f"test_{key}_{k}": v for k, v in res.items()})

    return results, wandb_data
