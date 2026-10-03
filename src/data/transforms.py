"""
Image preprocessing and augmentation.

Train:  mild random crop + flips + rotation + light color jitter.
Eval:   deterministic resize + center crop.

Why these choices for skin lesion classification:
  - Color IS the diagnostic signal (melanoma variegation) — keep color
    jitter light so we don't destroy the very feature we want the model
    to learn.
  - Lesions sit roughly in the center of the frame — RandomResizedCrop
    with scale >= 0.8 stops us cropping them out entirely.
  - Horizontal + vertical flips are safe: skin lesions have no
    orientation semantics (unlike, say, X-rays).
"""
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),          # 256 for 224
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])