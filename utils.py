def parse_wh(wh: str):
    assert isinstance(wh, str)
    splits = wh.split(",")
    splits = list(map(lambda x: int(x.strip()), splits))
    if len(splits) == 1:
        w = h = splits[0]
    elif len(splits) == 2:
        w, h = splits
    else:
        raise Exception(f"Invalid wh format, should be w,h but {wh}")
    return w, h


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
