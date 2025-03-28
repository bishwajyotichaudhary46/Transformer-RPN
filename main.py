import torch
from torch.utils.data.dataloader import DataLoader
from Transformer_RPN.utils.common import read_yaml
from Transformer_RPN.pipeline.load_data import LoadData
from Transformer_RPN.pipeline.custom_dataset import CustomDataset
from Transformer_RPN.pipeline.model import TransformerRPN
from Transformer_RPN.pipeline.train import Train
import random

data_ingestion_params = read_yaml('config/config.yaml')['data_ingestion']

classes = data_ingestion_params['class']
classes = sorted(classes)
classes = ['background'] + classes
label2idx = {classes[idx]: idx for idx in range(len(classes))}
idx2label = {idx: classes[idx] for idx in range(len(classes))}

load_data = LoadData(img_dir=data_ingestion_params['img_dir'], xml_dir=data_ingestion_params['xml_dir'], label2idx=label2idx)
data = load_data.load()

total_len_data = len(data)
train_size = int(total_len_data * 0.7)
val_size = int(total_len_data * 0.15)
test_size = total_len_data - train_size - val_size
print(f"Train Size: {train_size}\nValidation Size : {val_size}\nTest_size: {test_size}")


random.shuffle(data)
train_data = data[:train_size]
val_data = data[train_size: train_size+val_size]
test_data = data[train_size+val_size:]


train_datasets = CustomDataset(train_data)
test_datasets = CustomDataset(test_data)
val_datasets = CustomDataset(val_data)


train_dl = DataLoader(train_datasets,batch_size=1,shuffle=True,num_workers=4)
test_dl = DataLoader(test_datasets,batch_size=1,shuffle=True,num_workers=4)
val_dl = DataLoader(val_datasets,batch_size=1,shuffle=True,num_workers=4)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


rpn_params = read_yaml("params.yaml")['rpn_params']

roi_head_param = read_yaml('params.yaml')['roi_params']

model = TransformerRPN(roi_head_param=roi_head_param,
                        rpn_params=rpn_params,
                        device=device, training=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
loaders = {'train': train_dl, 'val': val_dl}

model = model.to(device)

model = Train(model=model, dataloaders=train_dl,optimizer=optimizer,num_epochs=50, device=device).train_model()

