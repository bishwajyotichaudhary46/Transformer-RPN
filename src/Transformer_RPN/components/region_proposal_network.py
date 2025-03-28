import torch
import torch.nn as nn
from Transformer_RPN.components.anchor_genrate import GenerateAnchor
from Transformer_RPN.components.proposal_prediction import ProposalPrediction

class RegionProposalNetwork(nn.Module):  

    def __init__(self,  ascpect_ratios, scales, in_channels=768 ):
        super(RegionProposalNetwork, self).__init__()
        self.aspect_ratios = ascpect_ratios
        self.scales = scales
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)
        
        
        # 3x3 conv layer
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        
        # 1x1 classification conv layer
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1)
        
        # 1x1 regression
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1, stride=1)
        
        for layer in [self.rpn_conv, self.cls_layer, self.bbox_reg_layer]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
        
        
    def forward(self,image, feature):
        # taking only features tokens 
        feature = feature[:,1:,:]
        # reshaping into [B, C, H, W]
        feature = feature.reshape(1, 768, 14, 14)
        rpn_feat = nn.ReLU()(self.rpn_conv(feature))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)
        
        #anchors generation
        anchors = self.generate_anchors = GenerateAnchor(self.scales, self.aspect_ratios).generate_anchors(image, feature)
        
        #gives the number anchors in each locations
        number_of_anchors_each_location = cls_scores.size(1)
        # changes axis of cls scores to (B, H, W, C)
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        # reshapes into (H*W*C, 1)
        cls_scores = cls_scores.reshape(-1, 1)
         
        # reshaping regression into (B, Number_of_anchors, 4, H, W)
        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            number_of_anchors_each_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1])
        
        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)
        box_transform_pred = box_transform_pred.reshape(-1, 4)
        
        # transform anchors box to prdiction box 
        proposals = ProposalPrediction().predict( box_transform_pred.detach().reshape(-1, 1, 4), anchors)


        return cls_scores, box_transform_pred, proposals