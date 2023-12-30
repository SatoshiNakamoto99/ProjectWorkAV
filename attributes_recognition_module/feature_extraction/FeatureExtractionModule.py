from torch import nn
import torch.nn.functional as F
import attributes_recognition_module.feature_extraction.ConvNeXt as ConvNeXt


class FeatureExtractionModule(nn.Module):
    def __init__(self, version = "base"):
        super(FeatureExtractionModule, self).__init__()
        
        if version == "base":
            self.convnext = ConvNeXt.convnext_base(pretrained=True, in_22k=False)
        elif version == "small":
            self.convnext = ConvNeXt.convnext_small(pretrained=True, in_22k=True)
        elif version == "tiny":
            self.convnext = ConvNeXt.convnext_tiny(pretrained=True, in_22k=True)
        elif version == "large":
            self.convnext = ConvNeXt.convnext_large(pretrained=True, in_22k=True)
        elif version == "xlarge":
            self.convnext = ConvNeXt.convnext_xlarge(pretrained=True, in_22k=True)
        else:
            raise Exception("Version not supported")
        
    def forward(self, x):
        return self.convnext.forward_features(x)
    


