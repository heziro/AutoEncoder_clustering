import albumentations as A
import torch

def augmentation(data,crop_size):
    data_list = []
    transform = A.Compose([A.RandomCrop(width=crop_size, height= crop_size),
                           A.HorizontalFlip(p=0.5),
                           A.RandomBrightnessContrast(p=0.2)])

    transform_2 = A.compose([A.augmentations.transforms.Blur(blur_limit=7, always_apply=False, p=0.5),
                             A.augmentations.transforms.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20,
                                                                  always_apply=False, p=0.5),
                             A.augmentations.transforms.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True,
                                                                    always_apply=False, p=0.5)])

    transform_3 = A.compose([A.augmentations.transforms.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8),
                                                               always_apply=False, p=0.5),A.HorizontalFlip(p=0.5)])

    for image in data:
        transformed = transform(image)['image']
        data_list.append(transformed)
        transformed_2 = transform_2(image)['image']
        data_list.append(transformed_2)
        transformed_3 = transform_3(image)['image']
        data_list.append(transformed_3)
    data_tensor = torch.Tensor(data_list)
    return data_tensor


