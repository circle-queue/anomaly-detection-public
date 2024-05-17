import contextlib
import random

import numpy as np
import torch
from rd4ad.dataset import MVTecDataset, get_data_transforms
from rd4ad.resnet_models import resnet18
from rd4ad.test import evaluation
from torch.nn import functional as F
from torchvision.datasets import ImageFolder


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def layers_loss(layers_in, layers_out) -> torch.Tensor:
    return sum(  # Sum batch losses across the layers
        [
            1 - F.cosine_similarity(in_.flatten(1), out.flatten(1))
            for in_, out in zip(layers_in, layers_out)
        ]
    )


def train(_class_):
    epochs = 200
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
    betas = (0.5, 0.999)
    train_path = f"./mvtec/{_class_}/train"
    test_path = f"./mvtec/{_class_}"
    ckp_path = f"./checkpoints/wres50_{_class_}.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(_class_, device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(
        root=test_path,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False
    )

    teacher_encoder, bn, student_decoder = [x.to(device) for x in resnet18()]
    teacher_encoder.eval()

    optimizer = torch.optim.Adam(
        list(student_decoder.parameters()) + list(bn.parameters()),
        lr=learning_rate,
        betas=betas,
    )

    for epoch in range(epochs):
        bn.train()
        student_decoder.train()
        total_loss = 0
        for img, label in train_dataloader:
            img = img.to(device)
            input_layers = teacher_encoder(img)
            output_layers = student_decoder(bn(input_layers))

            loss = layers_loss(input_layers, output_layers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
        print(
            f"epoch [{epoch+1}/{epochs}], loss:{total_loss/len(train_dataloader):.4f}",
            flush=True,
        )
        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(
                teacher_encoder, bn, student_decoder, test_dataloader, device
            )
            print(
                f"Pixel Auroc:{auroc_px:.3f}, Sample Auroc{auroc_sp:.3f}, Pixel Aupro{aupro_px:.3}"
            )

    torch.save(
        {"bn": bn.state_dict(), "decoder": student_decoder.state_dict()},
        ckp_path,
    )
    return auroc_px, auroc_sp, aupro_px


if __name__ == "__main__":
    setup_seed(111)
    item_list = [
        "carpet",
        "bottle",
        "hazelnut",
        "leather",
        "cable",
        "capsule",
        "grid",
        "pill",
        "transistor",
        "metal_nut",
        "screw",
        "toothbrush",
        "zipper",
        "tile",
        "wood",
    ]
    with open("output.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            for _class_ in item_list:
                train(_class_)
