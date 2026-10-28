"""Resolve historical model module names in trusted PyTorch checkpoints."""

import pickle


_MODEL_MODULES = {
    "model": "segmentation.model.model",
    "model_multiscale": "segmentation.model.model_multiscale",
    "model_multiscale_group": "segmentation.model.model_multiscale_group",
}


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Released ProtoSeg files were saved inside a proto_segmentation package.
        if module.startswith("proto_segmentation."):
            module = module[len("proto_segmentation."):]
        return super().find_class(_MODEL_MODULES.get(module, module), name)


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()
