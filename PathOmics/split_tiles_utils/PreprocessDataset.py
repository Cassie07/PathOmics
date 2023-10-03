import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

class PreprocessDataset(Dataset):
    def __init__(self, images, image_names):#, train = True):

        self.img_ls = images
        self.img_name_ls = image_names
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean = mean, std = std)
            ]
        )


    def __len__(self):
        return len(self.img_ls)

    def __getitem__(self, idx):
        img_name = self.img_name_ls[idx]
        image = self.img_ls[idx]

#         image = Image.open(img_name).convert("RGB")
        image = self.transform(image)

        return image, img_name