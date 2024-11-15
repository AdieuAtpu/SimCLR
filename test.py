import os
import torch
import torchvision
from models.resnet_simclr import ResNetSimCLR

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.multiprocessing.freeze_support()

    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)

    checkpoint_path = 'runs/Nov13_16-54-33_Eri/checkpoint_0010.pth.tar'

    # Configure logging

    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        print(k)

    for k in list(state_dict.keys()):
        if k.startswith('res_backbone.'):
            if k.startswith('res_backbone') and not k.startswith('res_backbone.fc'):
                state_dict[k[len("res_backbone."):]] = state_dict[k]
        del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']