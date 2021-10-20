import sys

import torch

sys.path.insert(1, 'invertinggradients')
from invertinggradients.inversefed.reconstruction_algorithms import GradientReconstructor 

from src.training import CrossEntropy

def get_gradient(model, input_data, gt_labels, use_vb, train_mode):
    ce = CrossEntropy()
    if train_mode:
        model.train()
    else:
        model.eval()

    model.zero_grad()
    if use_vb:
        target_loss = ce(model(input_data), gt_labels) + model.loss()
    else:
        target_loss = ce(model(input_data), gt_labels)
    gradient = torch.autograd.grad(target_loss, model.parameters(), allow_unused=True)
    gradient = [grad.detach() for grad in gradient]

    return gradient

def gradient_inversion(gradient, labels, model, data_shape, dm, ds, config):
    rec_machine = GradientReconstructor(model, (dm, ds), config, num_images=labels.shape[0])
    output, stats = rec_machine.reconstruct(gradient, labels, img_shape=data_shape)
    return output
