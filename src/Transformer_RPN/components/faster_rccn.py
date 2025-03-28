import torch
import torch.nn as nn
from Transformer_RPN.components.vision_transformer import VisionTransformer
from Transformer_RPN.components.region_proposal_network import RegionProposalNetwork
from Transformer_RPN.components.roi_head import ROIHead
from Transformer_RPN.utils.common import transform_boxes_to_original_size

class FasterRCNN(nn.Module):
    def __init__(self, model_config, num_classes, rpn_params, roi_head_param):
        super(FasterRCNN, self).__init__()
        self.model_config = model_config
        self.vision_model = VisionTransformer()
        self.rpn_model = RegionProposalNetwork(ascpect_ratios=rpn_params['ascpect_ratios'], scales=rpn_params['scales'], in_channels=rpn_params['input_channels'], rpn_prenms_topk=rpn_params['rpn_prenms_train_topk'], rpn_nms_threshold=rpn_params['rpn_nms_threshold'], rpn_topk=rpn_params['rpn_train_topk'], high_iou_threshold=rpn_params['high_iou_threshold'],low_iou_threshold=rpn_params['low_iou_threshold'])
        self.roi_head_model = ROIHead(model_config=roi_head_param, num_classes=roi_head_param['num_classes'],in_channels=768)
        
       
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']

        dtype, device = image.dtype, image.device
        
        # Normalize
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        #############
        
        # Resize to 1000x600 such that lowest size dimension is scaled upto 600
        # but larger dimension is not more than 1000
        # So compute scale factor for both and scale is minimum of these two
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(float(self.min_size) / min_size, float(self.max_size) / max_size)
        scale_factor = scale.item()
        
        # Resize image based on scale computed
        image = torch.nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )

        if bboxes is not None:
            # Resize boxes by
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                for s, s_orig in zip(image.shape[-2:], (h, w))
            ]
            ratio_height, ratio_width = ratios
            xmin, ymin, xmax, ymax = bboxes.unbind(2)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        return image, bboxes
    
    def forward(self, image, target=None):
        old_shape = image.shape[-2:]
        base_feature = self.vision_model(image)
        
        # Call RPN and get proposals
        rpn_output = self.rpn_model(image,base_feature, target)

        
        # Call ROI head and convert proposals to boxes
        frcnn_output = self.roi_head_model(base_feature, rpn_output['proposals'], image.shape[-2:], target)
        if not self.training:
            frcnn_output['boxes'] = transform_boxes_to_original_size(frcnn_output['boxes'],
                                                                     image.shape[-2:],
                                                                     old_shape)
        return rpn_output, frcnn_output