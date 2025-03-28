import torch
import math

class ProposalPrediction:
    def __init__(self):
        pass
        
    def predict(self, box_transform_pred, anchors):
   
        box_transform_pred = box_transform_pred.reshape(
            box_transform_pred.size(0), -1, 4)
        
        w = anchors[:, 2] - anchors[:, 0]
        h = anchors[:, 3] - anchors[:, 1]

        center_x = anchors[:, 0] + 0.5 * w
        center_y = anchors[:, 1] + 0.5 * h
        
        dx = box_transform_pred[..., 0]
        dy = box_transform_pred[..., 1]

        dw = box_transform_pred[..., 2]
        dh = box_transform_pred[..., 3]
       
        
        
        dw = torch.clamp(dw, max=math.log(1000.0 / 16))
        dh = torch.clamp(dh, max=math.log(1000.0 / 16))
        
        pred_center_x = dx * w[:, None] + center_x[:, None]
        pred_center_y = dy * h[:, None] + center_y[:, None]
        pred_w = torch.exp(dw) * w[:, None]
        pred_h = torch.exp(dh) * h[:, None]
        
        pred_box_x1 = pred_center_x - 0.5 * pred_w
        pred_box_y1 = pred_center_y - 0.5 * pred_h
        pred_box_x2 = pred_center_x + 0.5 * pred_w
        pred_box_y2 = pred_center_y + 0.5 * pred_h
        
        pred_boxes = torch.stack((
            pred_box_x1,
            pred_box_y1,
            pred_box_x2,
            pred_box_y2),
            dim=2)
        
        return pred_boxes
        
        
        