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

def resize_and_cut_image(image: Image, size) -> Image:
    width, height = size
    ori_width, ori_height = image.size
    origin_ratio = ori_height / ori_width
    target_ratio = height / width

    if origin_ratio > target_ratio: # 세로가 더 긴 경우
        new_width = width
        new_height = int(width * origin_ratio)
        pixel_to_cut = (new_height - height) / 2

        image = image.resize((new_width, new_height), Image.BICUBIC)
        image = image.crop((0, pixel_to_cut, new_width, new_height - pixel_to_cut))

    else: # 가로가 더 긴 경우 
        new_width = int(height / origin_ratio)
        new_height = height
        pixel_to_cut = (new_width - width) / 2

        image = image.resize((new_width, new_height), Image.BICUBIC)
        image = image.crop((pixel_to_cut, 0, new_width - pixel_to_cut, new_height))

    return image

def shuffle_two_image_patches(image1_path, image2_path, alpha=0.5, patch_rows=5, patch_cols=5, seed=None):
    if seed is not None:
        # for deterministic sample
        random.seed(seed)
    
    image1 = Image.open(image1_path).convert("RGBA")
    image2 = Image.open(image2_path).convert("RGBA")

    width, height = image1.size

    image2 = resize_and_cut_image(image=image2, size=image1.size)

    # 패치의 크기 계산
    patch_width = width // patch_cols
    patch_height = height // patch_rows

    patch_num = patch_rows * patch_cols

    # 패치를 리스트로 자르기
    patches = [[],[]]
    for row in range(patch_rows):
        for col in range(patch_cols):
            left = col * patch_width
            upper = row * patch_height
            right = left + patch_width
            lower = upper + patch_height
            patch1 = image1.crop((left, upper, right, lower))
            patches[0].append(patch1)

            patch2 = image2.crop((left, upper, right, lower))
            patches[1].append(patch2)

    num_benign = int(patch_num * alpha)
    benign_idx = random.sample(range(0, patch_num), num_benign)

    # 새 이미지 생성
    new_image = Image.new('RGB', (width, height))

    # 패치를 다시 합치기
    idx = 0
    for row in range(patch_rows):
        for col in range(patch_cols):
            left = col * patch_width
            upper = row * patch_height
            if idx not in benign_idx: new_image.paste(patches[0][idx], (left, upper))
            else: new_image.paste(patches[1][idx], (left, upper))

            idx += 1
    
    return new_image


#####################################

def test(image_path, image_path2, size=None):
    # image = Image.open(image_path)
    new_image = shuffle_two_image_patches(image_path, image_path2, alpha=0.3)
    new_image.save('test.png')

# 사용 예시

if __name__=='__main__':
    test('datasets/AdvBenchM/images/harmful/bomb_explosive/1.jpg', 
        'datasets/AdvBenchM/images/harmless/apple.png')
    

    # shuffle_image_patches('harmful\\bomb_explosive\\1.jpg')
    # shuffle_two_image_patches('harmful\\bomb_explosive\\1.jpg', 'harmless\\apple.png')


