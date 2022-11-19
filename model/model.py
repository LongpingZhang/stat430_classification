import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes, loss_fn, device, threshold):
        super().__init__()
        self.pretrained_alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        self.num_classes = num_classes
        self.pretrained_alexnet.classifier[6] = nn.Linear(4096, self.num_classes)
        self.loss_fn = loss_fn
        self.device = device
        self.threshold = threshold
        
    def forward(self, img):
        return self.pretrained_alexnet(img)
    
    def training_step(self, batch):
        imgs, labels = batch
        imgs = imgs.float().to(self.device)
        labels = labels.float().to(self.device)
        
        preds = self.forward(imgs)
        return self.loss_fn(
                   torch.flatten(preds), torch.flatten(labels)
               )
    
    def _getloss(self, dataloader):
        preds_class_list = []
        labels_list = []
        
        
        cum_loss = 0
        cum_batches = 0
        for batch in dataloader:
            imgs, labels = batch
            imgs = imgs.float().to(self.device)
            labels = labels.float().to(self.device)
            preds = self.forward(imgs)
            cum_loss += self.loss_fn(
                torch.flatten(preds), torch.flatten(labels)
            )
            cum_batches += 1
            
            preds_prob = 1 / (1 + torch.exp(-preds))
            preds_class = (preds_prob > self.threshold).float()
            preds_class_list.extend(preds_class.data.cpu().numpy().flatten())
            labels_list.extend(labels.data.cpu().numpy().flatten())
            
        loss = cum_loss / cum_batches
        
        cf_matrix = confusion_matrix(labels_list, preds_class_list)
        return loss, cf_matrix
    
    def validation_step(self, val_dataloader):
        return self._getloss(val_dataloader)
    
    def testing_step(self, test_dataloader):
        return self._getloss(test_dataloader)
    
    