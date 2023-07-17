import albumentations as albu


def get_training_augmentation(image_size,type = 1):
    if type == 1:

        train_transform = [
            albu.Resize(image_size, image_size),
            albu.HorizontalFlip(p=0.2),
            albu.VerticalFlip(p=0.2),
            albu.RandomRotate90(p=0.2),
        ]
    else:
        train_transform = [
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(p=0.2),
                ]
            ),
        ]

    return albu.Compose(train_transform)


def get_validation_augmentation(image_size):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(image_size, image_size)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
