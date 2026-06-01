import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, input_size=224, batch_size=16, num_workers=0):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ds.classes
