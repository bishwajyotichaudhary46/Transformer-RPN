import torch

class GenerateAnchor:
    def __init__(self, scales, aspect_ratios):
        self.scales = scales
        self.aspect_ratios = aspect_ratios

    def generate_anchors(self, image, featuremap):

        grid_h, grid_w = featuremap.shape[-2:]
        image_h, image_w = image.shape[-2:]

        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=featuremap.device)
        stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=featuremap.device)
        
        scales = torch.as_tensor(self.scales, dtype=featuremap.dtype, device=featuremap.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=featuremap.dtype, device=featuremap.device)
        
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()
        
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=featuremap.device) * stride_w
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=featuremap.device) * stride_h
        
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)

        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)
       
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))

        anchors = anchors.reshape(-1, 4)
        
        return anchors
    