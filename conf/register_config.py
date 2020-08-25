import re
import conf.augmentations as aug
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')

def camel2snake(name: str):
    "Convert CamelCase to snake_case"
    s1   = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def formulate_name(classname: str):
    "Formulate 'name' for config dataclass"
    snake_case = camel2snake(classname)
    name = re.sub('_conf','',snake_case)
    return name

# register augmentations
for conf in aug.configs:
    name = formulate_name(conf.__name__)
    cs.store(group='aug',name=name,node=conf)