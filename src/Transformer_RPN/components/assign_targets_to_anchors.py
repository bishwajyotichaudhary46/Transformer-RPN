from torchvision.ops import box_iou
import torch
class AssignTarget:
    def __init__(self, anchors, gt_boxes, low_iou_threshold, high_iou_threshold):
        self.anchors = anchors
        self.gt_boxes = gt_boxes
        self.low_iou_threshold = low_iou_threshold
        self.high_iou_threshold = high_iou_threshold


    def assign_targets_to_anchors(self):
    
    
        iou_matrix = box_iou(self.gt_boxes, self.anchors)
        
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
    
        best_match_gt_idx_pre_thresholding = best_match_gt_idx.clone()
        
        
        below_low_threshold = best_match_iou < self.low_iou_threshold
        between_thresholds = (best_match_iou >= self.low_iou_threshold) & (best_match_iou < self.high_iou_threshold)
        best_match_gt_idx[below_low_threshold] = -1
        best_match_gt_idx[between_thresholds] = -2
        
        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1)

        gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])
    
        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        
        best_match_gt_idx[pred_inds_to_update] = best_match_gt_idx_pre_thresholding[pred_inds_to_update]
        
        
        matched_gt_boxes = self.gt_boxes[best_match_gt_idx.clamp(min=0)]
        
        labels = best_match_gt_idx >= 0
        labels = labels.to(dtype=torch.float32)
        
        background_anchors = best_match_gt_idx == -1
        labels[background_anchors] = 0.0
        
        ignored_anchors = best_match_gt_idx == -2
        labels[ignored_anchors] = -1.0
        
        return labels, matched_gt_boxes
