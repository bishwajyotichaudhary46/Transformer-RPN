import torch
import torch.nn as nn
from Transformer_RPN.components.vision_transformer import VisionTransformer
from Transformer_RPN.components.region_proposal_network import RegionProposalNetwork
from Transformer_RPN.components.roi_head import ROIHead
from Transformer_RPN.utils.common import transform_boxes_to_original_size

class TransformerRPN(nn.Module):
    def __init__(self, rpn_params, roi_head_param,device, training=None):
        super(TransformerRPN, self).__init__()

        self.vision_model = VisionTransformer()

        self.rpn_model = RegionProposalNetwork(ascpect_ratios=rpn_params['ascpect_ratios'],
                                                scales=rpn_params['scales'],
                                                in_channels=rpn_params['input_channels'],
                                                rpn_prenms_topk=rpn_params['rpn_prenms_train_topk'],
                                                rpn_nms_threshold=rpn_params['rpn_nms_threshold'],
                                                rpn_topk=rpn_params['rpn_train_topk'],
                                                high_iou_threshold=rpn_params['high_iou_threshold'],
                                                low_iou_threshold=rpn_params['low_iou_threshold'])
        
        self.roi_head_model = ROIHead(model_config=roi_head_param,
                                       num_classes=roi_head_param['num_classes'],in_channels=768)
        
        self.training = training
        # self.image_mean = [0.485, 0.456, 0.406]
        # self.image_std = [0.229, 0.224, 0.225]
        # self.min_size = model_config['min_im_size']
        # self.max_size = model_config['max_im_size']
        
    
    def forward(self, image,device, target=None):
        old_shape = image.shape[-2:]
        image = image.to(device)
        base_feature = self.vision_model(image)
        
        # Call RPN and get proposals
        rpn_output = self.rpn_model(image, base_feature, target=target, device=device)

        
        # Call ROI head and convert proposals to boxes
        frcnn_output = self.roi_head_model(base_feature,
                                            rpn_output['proposals'],
                                            image.shape[-2:],
                                            target, device)
        if not self.training:
            frcnn_output['boxes'] = transform_boxes_to_original_size(frcnn_output['boxes'],
                                                                     image.shape[-2:],
                                                                     old_shape)
        return rpn_output, frcnn_output