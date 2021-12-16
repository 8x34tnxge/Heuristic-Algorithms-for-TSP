from yacs.config import CfgNode


config = CfgNode(new_allowed=True)
config.merge_from_file("./cityInfo.yaml")
config.freeze()


def getConfig() -> CfgNode:
    return config
