from collections import defaultdict
import torch
import numpy as np

class CrossEntropy:
    def __init__(self):
        self.metric_fn = torch.nn.CrossEntropyLoss()
        self.format = '.5f'
        self.name = 'CrossEntropy'
    
    def __call__(self, prediction, target):
        return self.metric_fn(prediction, target)

class Accuracy:
    def __init__(self):
        self.metric_fn = self.acc
        self.format = '.2%'
        self.name = 'Accuracy'
    
    def acc(self, prediction, target):
        value = (prediction.data.argmax(dim=1) == target).sum().float() / target.shape[0]
        return value.detach()

    def __call__(self, prediction, target):
        return self.metric_fn(prediction, target)

def train(model, optimizer, train_data, test_data, epochs, vb, device):
    ce_loss = CrossEntropy()
    acc = Accuracy()
    
    metrics_dict = defaultdict(list)
    print('START TRAINING')
    for epoch in range(epochs):
        model.train()
        batch_metrics_dict = defaultdict(list)
        for inputs, targets in train_data:
            optimizer.zero_grad()

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if vb:
                loss = ce_loss(outputs, targets) + model.loss() # add the VB loss to the overall loss
            else:
                loss = ce_loss(outputs, targets)

            loss.backward()
            optimizer.step()
            
            #calculate all metrics
            batch_metrics_dict[acc.name+'_TRN'] = acc(outputs, targets).item()
            batch_metrics_dict[ce_loss.name+'_TRN'].append(loss.item())
        

        model.eval()
        for inputs, targets in test_data:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = ce_loss(outputs, targets)
            if vb:
                loss = loss + model.loss() # add the VB loss to the overall loss

            batch_metrics_dict[acc.name+'_TST'] = acc(outputs, targets).item()
            batch_metrics_dict[ce_loss.name+'_TST'].append(loss.item())

        status_str = f'Epoch {epoch+1:3}: '
        for k, v in batch_metrics_dict.items():
            epoch_mean = np.mean(v)
            metrics_dict[k].append(epoch_mean)
            status_str += f'{k}: {epoch_mean:.5f} | '

        print(status_str[:-2])
    print('DONE WITH TRAINING')
    return metrics_dict