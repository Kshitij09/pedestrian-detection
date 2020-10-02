from omegaconf import OmegaConf
import collections
def collate_fn(batch):
    return tuple(zip(*batch))

# Borrowed from
# https://github.com/Erlemar/wheat/blob/54d253cc7ff559bebc1056dca8b5b058fb75fc9f/src/utils/utils.py#L110
def flatten_omegaconf(d, sep="_"):

    d = OmegaConf.to_container(d)

    obj = collections.OrderedDict()

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    obj = {k: v for k, v in obj.items() if type(v) in [int, float]}
    # obj = {k: v for k, v in obj.items()}

    return obj
