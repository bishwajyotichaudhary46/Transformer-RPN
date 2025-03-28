import time
import torch
from tqdm import tqdm
import numpy as np

class Train:
    def __init__(self, model, dataloaders, optimizer, num_epochs, device):
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def train_model(self):
        mean = torch.as_tensor(self.image_mean, device=self.device)
        std = torch.as_tensor(self.image_std,  device=self.device)

        acc_steps = 1

        step_count = 1

        for i in range(self.num_epochs):
            rpn_classification_losses = []
            rpn_localization_losses = []
            frcnn_classification_losses = []
            frcnn_localization_losses = []
            self.optimizer.zero_grad()
            
            for image, target, fname in tqdm(self.dataloaders):
                image = image.to(self.device)
                image = (image - mean[:, None, None]) / std[:, None, None]
                rpn_output, frcnn_output = self.model(image = image, target=target, device=self.device)
                
                rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
                frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
                loss = rpn_loss + frcnn_loss
                
                rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
                rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
                frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
                frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())
                loss = loss / acc_steps
                loss.backward()
                if step_count % acc_steps == 0:
                    self.optimizer.step()
                    
                step_count += 1
            print('Finished epoch {}'.format(i))

            # torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
            #                                                         train_config['ckpt_name']))
            loss_output = ''
            loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
            loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
            loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
            loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
            print(loss_output)
        print('Done Training...')
        return self.model
