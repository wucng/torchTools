import json
from .defaults import cfg

def get_cfg():
    return cfg.copy()

def merge_from_cfg(cfg,cfg2):
    for k,v in cfg2.items():
        if k not in cfg:continue
        for k1,v1 in v.items():
            if k1 not in cfg[k]: continue
            for k2,v2 in v1.items():
                if k2 not in cfg[k][k1]: continue
                cfg[k][k1][k2] = v2

    # cfg.update(cfg2)
    return cfg

def merge_from_file(cfg,filePath):
    cfg2 = json.load(open(filePath,'r'))
    return merge_from_cfg(cfg,cfg2)

