import torch
from Transformer_RPN.utils.common import clamp_boxes_to_image_boundary
class FilterProposal:
    def __init__(self, proposals, cls_scores, image_shape, rpn_prenms_topk, rpn_nms_threshold, rpn_topk):
        self.proposals = proposals
        self.cls_scores = cls_scores
        self.image_shape = image_shape
        self.rpn_prenms_topk = rpn_prenms_topk
        self.rpn_nms_threshold = rpn_nms_threshold
        self.rpn_topk = rpn_topk


    def filter_proposals(self):
    
        cls_scores = self.cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)
        _, top_n_idx = cls_scores.topk(min(self.rpn_prenms_topk, len(cls_scores)))
        
        cls_scores = cls_scores[top_n_idx]
        proposals = self.proposals[top_n_idx]
        
        proposals = clamp_boxes_to_image_boundary(proposals, self.image_shape)
        
        min_size = 16
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]
        
        
        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals, cls_scores, self.rpn_nms_threshold)
        keep_mask[keep_indices] = True
        keep_indices = torch.where(keep_mask)[0]
        # Sort by objectness
        post_nms_keep_indices = keep_indices[cls_scores[keep_indices].sort(descending=True)[1]]
        
        # Post NMS topk filtering
        proposals, cls_scores = (proposals[post_nms_keep_indices[:self.rpn_topk]],
                                 cls_scores[post_nms_keep_indices[:self.rpn_topk]])
        
        
        return proposals, cls_scores