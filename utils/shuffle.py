from PIL import Image
import random

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

def shuffle_image_patches(image_path, patch_rows=3, patch_cols=3, seed=None):
    # 이미지 열기
    if type(image_path) == str:
            image = Image.open(image_path).convert("RGB")
    if seed is not None:
        # for deterministic sample
        random.seed(seed)

    image = resize_image_to_longest_axis(image)
    width, height = image.size

    # 패치의 크기 계산
    patch_width = width // patch_cols
    patch_height = height // patch_rows

    # 패치를 리스트로 자르기
    patches = []
    for row in range(patch_rows):
        for col in range(patch_cols):
            left = col * patch_width
            upper = row * patch_height
            right = left + patch_width
            lower = upper + patch_height
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)

    # 패치 무작위 섞기
    random.shuffle(patches)

    # 새 이미지 생성
    new_image = Image.new('RGB', (width, height))

    # 패치를 다시 합치기
    idx = 0
    for row in range(patch_rows):
        for col in range(patch_cols):
            left = col * patch_width
            upper = row * patch_height
            new_image.paste(patches[idx], (left, upper))
            idx += 1
    return new_image

    # 이미지 저장 또는 반환

    # print(new_image.size)
    # new_image.show()

    # new_image.save(save_path)
    # print(f"Shuffled image saved to: {save_path}")
    # return new_image

if __name__=='__main__':
    shuffle_image_patches('harmful\\bomb_explosive\\1.jpg')
    # shuffle_two_image_patches('harmful\\bomb_explosive\\1.jpg', 'harmless\\apple.png')

