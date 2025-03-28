import torch
import torch.nn as nn
from Transformer_RPN.components.anchor_genrate import GenerateAnchor
from Transformer_RPN.components.proposal_prediction import ProposalPrediction
from Transformer_RPN.components.filter_proposal import FilterProposal
from Transformer_RPN.components.assign_targets_to_anchors import AssignTarget
from Transformer_RPN.utils.common import boxes_to_transformation_targets, sample_positive_negative
class RegionProposalNetwork(nn.Module):  

    def __init__(self,  ascpect_ratios, scales,  rpn_prenms_topk, rpn_nms_threshold, rpn_topk, low_iou_threshold, high_iou_threshold,in_channels=768):
        super(RegionProposalNetwork, self).__init__()
        self.aspect_ratios = ascpect_ratios
        self.scales = scales
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)
        
        self.rpn_prenms_topk = rpn_prenms_topk
        self.rpn_nms_threshold = rpn_nms_threshold
        self.rpn_topk = rpn_topk

        self.low_iou_threshold = low_iou_threshold
        self.high_iou_threshold = high_iou_threshold

        self.rpn_pos_count = int( 0.5 * 256)
        
        # 3x3 conv layer
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        
        # 1x1 classification conv layer
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1)
        
        # 1x1 regression
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1, stride=1)
        
        for layer in [self.rpn_conv, self.cls_layer, self.bbox_reg_layer]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
        
        
    def forward(self,image, feature,device, target=None):
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
        # reshape proposal into (N, 4)
        proposals = proposals.reshape(proposals.size(0), 4)
        proposals, scores = FilterProposal(proposals, cls_scores.detach(), image.shape, self.rpn_prenms_topk, self.rpn_nms_threshold, self.rpn_topk ).filter_proposals()
        rpn_output = {
            'proposals': proposals,
            'scores': scores
        }

        if not self.training or target is None:
            return rpn_output
        else:
            labels_for_anchors, matched_gt_boxes_for_anchors = AssignTarget(anchors = anchors, gt_boxes = target['bboxes'][0].to(device), low_iou_threshold = self.low_iou_threshold, high_iou_threshold = self.high_iou_threshold).assign_targets_to_anchors()

            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_anchors, anchors)
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels_for_anchors,
                positive_count=self.rpn_pos_count,
                total_count = 256)
            
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
            
            localization_loss = (
                    torch.nn.functional.smooth_l1_loss(
                        box_transform_pred[sampled_pos_idx_mask],
                        regression_targets[sampled_pos_idx_mask],
                        beta=1 / 9,
                        reduction="sum",
                    )
                    / (sampled_idxs.numel())
            ) 

            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_scores[sampled_idxs].flatten(),
                                                                            labels_for_anchors[sampled_idxs].flatten())

            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = localization_loss
            return rpn_output
            