from torchvision import transforms


BASIC_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.1, 0.6), contrast=1, saturation=0, hue=0.4),
        transforms.RandomRotation(degrees=30),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)
