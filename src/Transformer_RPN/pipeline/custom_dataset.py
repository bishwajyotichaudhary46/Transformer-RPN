from torch.utils.data.dataset import Dataset
import torchvision
from PIL import Image
import torch
class CustomDataset(Dataset):
    def __init__(self, data):
        self.target_size = (224,224)
        self.images_info = data
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.target_size),  # Resize images
            torchvision.transforms.ToTensor()  # Convert to tensor
        ])
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        img_info = self.images_info[index]
        img = Image.open(img_info['image'])
        
        # Store original dimensions
        original_w, original_h = img.size
        target_w, target_h = self.target_size
    
        # Resize the image
        img_tensor = self.transform(img)
        
        # Scale bounding boxes to new image size
        targets = {}
        targets['bboxes'] = []
        targets['labels'] = torch.as_tensor([d['label'] for d in img_info['detections']], dtype=torch.int64)

        scale_x = target_w / original_w
        scale_y = target_h / original_h

        for detection in img_info['detections']:
            x1, y1, x2, y2 = detection['bbox']

            # Scale bbox to new dimensions
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            targets['bboxes'].append([x1, y1, x2, y2])

        targets['bboxes'] = torch.as_tensor(targets['bboxes'], dtype=torch.float32)

        return img_tensor, targets, img_info['image']