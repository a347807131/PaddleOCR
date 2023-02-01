import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="paddle-ocr",

    # track hyperparameters and run metadata
    config={
        "model_name": "chinese_cht_PP-OCRv3_rec_train"
    }
)


def log_stats(strs: str):
    ## added by gatsby

    if strs is not None:
        str_splits = strs.split(',')
        state_dict = {}
        for split in str_splits:
            try:
                kv = split.split(':')
                if len(kv) == 2: state_dict[kv[0]] = float(kv[1])
            except:
                continue
        wandb.log(state_dict)
