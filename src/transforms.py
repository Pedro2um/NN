from torchvision import transforms


def test_transform():
 return transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def vannila_transform():
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)), 
            transforms.ToTensor()])

# deve ser usada (obrigatoriamente) depois do vannila_transform
def train_transform():
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=5,                    # rotação no intervalo [-5°, +5°]
            translate=(0.05, 0.05)        # até 5% de deslocamento em x e y
        ),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# deve ser usada (obrigatoriamente) depois do vannila_transform
def val_transform():
    return transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
