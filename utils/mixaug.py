"""
JOOD
Copyright (c) 2025-present NAVER Corp.
Apache License v2.0
"""
import numpy as np
from PIL import Image

def resize_image_to_longest_axis(image, target_longest_side=320):
    """
    Resize an image such that its longest side becomes the target length while preserving aspect ratio.

    :param image: The image to resize
    :param target_longest_side: The target length for the longest side
    :return: The resized image
    """
    width, height = image.size
    if width > height:
        new_width = target_longest_side
        new_height = int((target_longest_side / width) * height)
    else:
        new_height = target_longest_side
        new_width = int((target_longest_side / height) * width)
    return image.resize((new_width, new_height), Image.BICUBIC)


def pad_image(image, target_size):
    """
    Pad the image equally on all sides to match the target size.

    :param image: The image to pad
    :param target_size: A tuple specifying the target size (width, height)
    :return: The padded image
    """
    new_image = Image.new("RGBA", target_size, (255, 255, 255, 0))
    paste_position = ((target_size[0] - image.size[0]) //
                      2, (target_size[1] - image.size[1]) // 2)
    new_image.paste(image, paste_position)
    return new_image

def mixup_images(image1_path, image2_path, alpha=0.5):
    """
    Mixes two images with a given alpha value and returns the result in base64 format.

    :param image1_path: Path to the first image
    :param image2_path: Path to the second image
    :param alpha: Blending factor. 0.0 means only image1, 1.0 means only image2.
    :return: Base64 encoded string of the mixed image
    """
    # Open the images
    image1 = Image.open(image1_path).convert("RGBA")
    image2 = Image.open(image2_path).convert("RGBA")

    # Resize images such that the longest side becomes 320 while preserving aspect ratio
    image1 = resize_image_to_longest_axis(image1)
    image2 = resize_image_to_longest_axis(image2)

    # Determine the target size based on the largest dimensions of the resized images
    target_width = max(image1.size[0], image2.size[0])
    target_height = max(image1.size[1], image2.size[1])
    target_size = (target_width, target_height)

    # Pad images to the target size
    image1 = pad_image(image1, target_size)
    image2 = pad_image(image2, target_size)

    # Blend the images
    mixed_image = Image.blend(image1, image2, alpha)
    return mixed_image

def cutmix_resizemix_images(image1_path, image2_path, alpha=0.5):
    """
    put image 1 to image 2, with alpha portion (for each length of axis)

    :param image1_path: Path to the first image
    :param image2_path: Path to the second image
    """
    # Open the images
    image1 = Image.open(image1_path).convert("RGBA")
    image2 = Image.open(image2_path).convert("RGBA")

    # Resize images such that the longest side becomes 320 while preserving aspect ratio
    image1 = resize_image_to_longest_axis(image1)
    image2 = resize_image_to_longest_axis(image2)

    # Determine the target size based on the largest dimensions of the resized images
    target_width = max(image1.size[0], image2.size[0])
    target_height = max(image1.size[1], image2.size[1])
    target_size = (target_width, target_height)

    # Pad images to the target size
    image1 = pad_image(image1, target_size)
    image2 = pad_image(image2, target_size)

    if alpha == 0:
        return image1
    elif alpha == 1:
        return image2

    # Get dimensions
    width, height = image1.size
    
    # Calculate mask dimensions based on alpha
    mask_width = int(width * alpha)
    mask_height = int(height * alpha)
    
    # Create a random position for the patch
    x = np.random.randint(0, width - mask_width)
    y = np.random.randint(0, height - mask_height)

    #Crop the patch from image1
    patch = image2.resize((mask_width, mask_height))

    # Create a copy of image2 and paste the patch onto it
    mixed_image = image1.copy()
    mixed_image.paste(patch, (x, y))

    return mixed_image

def cutmix_original_images(image1_path, image2_path, alpha=0.5):
    """
    put image 1 to image 2, with alpha portion (for each length of axis)

    :param image1_path: Path to the first image
    :param image2_path: Path to the second image
    """
    # Open the images
    image1 = Image.open(image1_path).convert("RGBA")
    image2 = Image.open(image2_path).convert("RGBA")

    # Resize images such that the longest side becomes 320 while preserving aspect ratio
    image1 = resize_image_to_longest_axis(image1)
    image2 = resize_image_to_longest_axis(image2)

    # Determine the target size based on the largest dimensions of the resized images
    target_width = max(image1.size[0], image2.size[0])
    target_height = max(image1.size[1], image2.size[1])
    target_size = (target_width, target_height)

    # Pad images to the target size
    image1 = pad_image(image1, target_size)
    image2 = pad_image(image2, target_size)

    if alpha == 0:
        return image1
    elif alpha == 1:
        return image2

    # Get dimensions
    width, height = image1.size
    
    # Calculate mask dimensions based on alpha
    mask_width = int(width * alpha)
    mask_height = int(height * alpha)
    
    # Create a random position for the patch
    x = np.random.randint(0, width - mask_width)
    y = np.random.randint(0, height - mask_height)

    # Crop the patch from image2
    patch = image2.crop((x, y, x + mask_width, y + mask_height))

    # Create a copy of image2 and paste the patch onto it
    mixed_image = image1.copy()
    mixed_image.paste(patch, (x, y))

    return mixed_image

def cutmixup_images(image1_path, image2_path, alpha=0.5, beta=0.5):
    """
    put image 1 to image 2, with alpha portion (for each length of axis)

    :param image1_path: Path to the first image
    :param image2_path: Path to the second image
    """
    # Open the images
    image1 = Image.open(image1_path).convert("RGBA")
    image2 = Image.open(image2_path).convert("RGBA")

    # Resize images such that the longest side becomes 320 while preserving aspect ratio
    image1 = resize_image_to_longest_axis(image1)
    image2 = resize_image_to_longest_axis(image2)

    # Determine the target size based on the largest dimensions of the resized images
    target_width = max(image1.size[0], image2.size[0])
    target_height = max(image1.size[1], image2.size[1])
    target_size = (target_width, target_height)

    # Pad images to the target size
    image1 = pad_image(image1, target_size)
    image2 = pad_image(image2, target_size)

    if alpha == 0:
        return image1
    elif alpha == 1:
        return image2

    # Get dimensions
    width, height = image1.size
    
    # Calculate mask dimensions based on alpha
    mask_width = int(width * alpha)
    mask_height = int(height * alpha)
    
    # Create a random position for the patch
    x = np.random.randint(0, width - mask_width)
    y = np.random.randint(0, height - mask_height)

    # Crop the patch from image2
    patch = image2.resize((mask_width, mask_height))
    mask = Image.new("L", patch.size, int(beta * 255))

    # Create a copy of image2 and paste the patch onto it
    mixed_image = image1.copy()
    mixed_image.paste(patch, (x, y), mask)

    return mixed_image