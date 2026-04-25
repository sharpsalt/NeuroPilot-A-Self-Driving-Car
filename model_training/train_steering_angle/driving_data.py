import os
import random

import cv2
import numpy as np

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/driving_dataset"))
DATA_FILE = os.path.join(BASE_PATH, "data.txt")

# Preprocessing and augmentation settings
BOTTOM_CROP = 150
TARGET_WIDTH = 200
TARGET_HEIGHT = 66
MAX_TRANSLATION_X = 40
MAX_TRANSLATION_Y = 8
STEERING_PER_PIXEL = 0.002
TURN_ABS_THRESHOLD = np.deg2rad(6.0)
TURN_SAMPLE_PROB = 0.45

# Data quality settings
MAX_ABS_STEERING_DEG = 75.0
ROBUST_Z_THRESHOLD = 6.0
LABEL_SMOOTH_WINDOW = 7
LABEL_SMOOTH_STRENGTH = 0.35

xs = []
ys = []

train_batch_pointer = 0
val_batch_pointer = 0


def _load_dataset():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Could not find driving labels file: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        for raw_line in file:
            parts = raw_line.strip().split()
            if len(parts) < 2:
                continue

            image_file = parts[0]
            angle_token = parts[1].split(",")[0]
            try:
                angle_deg = float(angle_token)
            except ValueError:
                continue

            xs.append(os.path.join(BASE_PATH, image_file))
            ys.append(angle_deg * np.pi / 180.0)

    if not xs:
        raise RuntimeError(f"No valid training samples found in: {DATA_FILE}")


def _shuffle_pairs(image_paths, angles):
    pairs = list(zip(image_paths, angles))
    random.shuffle(pairs)
    shuffled_xs, shuffled_ys = zip(*pairs)
    return list(shuffled_xs), list(shuffled_ys)


def _extract_frame_index(image_path):
    name = os.path.splitext(os.path.basename(image_path))[0]
    try:
        return int(name)
    except ValueError:
        return name


def _sort_temporal(image_paths, angles):
    pairs = list(zip(image_paths, angles))
    pairs.sort(key=lambda pair: _extract_frame_index(pair[0]))
    sorted_xs, sorted_ys = zip(*pairs)
    return list(sorted_xs), list(sorted_ys)


def _filter_missing_and_outliers(image_paths, angles):
    cleaned_paths = []
    cleaned_angles = []

    for path, angle in zip(image_paths, angles):
        if not os.path.exists(path):
            continue
        cleaned_paths.append(path)
        cleaned_angles.append(angle)

    if not cleaned_paths:
        raise RuntimeError("No valid image paths found after checking file existence")

    angle_array = np.array(cleaned_angles, dtype=np.float32)
    abs_limit = np.deg2rad(MAX_ABS_STEERING_DEG)
    abs_mask = np.abs(angle_array) <= abs_limit

    median = np.median(angle_array)
    mad = np.median(np.abs(angle_array - median))
    if mad < 1e-6:
        robust_mask = np.ones_like(abs_mask, dtype=bool)
    else:
        robust_z = 0.6745 * (angle_array - median) / mad
        robust_mask = np.abs(robust_z) <= ROBUST_Z_THRESHOLD

    keep_mask = np.logical_and(abs_mask, robust_mask)
    filtered_paths = [path for path, keep in zip(cleaned_paths, keep_mask) if keep]
    filtered_angles = [float(angle) for angle, keep in zip(angle_array, keep_mask) if keep]

    removed = len(image_paths) - len(filtered_paths)
    print(f"Filtered {removed} low-quality/outlier samples; kept {len(filtered_paths)}")

    if not filtered_paths:
        raise RuntimeError("All samples were filtered out; relax outlier thresholds")

    return filtered_paths, filtered_angles


def _smooth_steering_labels(angles):
    if len(angles) < 3:
        return angles

    window = LABEL_SMOOTH_WINDOW
    if window % 2 == 0:
        window += 1
    if len(angles) < window:
        return angles

    raw = np.array(angles, dtype=np.float32)
    half = window // 2

    rising = np.arange(1, half + 2, dtype=np.float32)
    kernel = np.concatenate([rising, rising[-2::-1]])
    kernel /= kernel.sum()

    padded = np.pad(raw, (half, half), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    blended = (1.0 - LABEL_SMOOTH_STRENGTH) * raw + LABEL_SMOOTH_STRENGTH * smoothed
    return blended.astype(np.float32).tolist()


def _crop_resize_normalize(image):
    image = image[-BOTTOM_CROP:]
    image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    return image.astype(np.float32) / 255.0


def _random_translate(image, angle):
    height, width = image.shape[:2]
    tx = random.uniform(-MAX_TRANSLATION_X, MAX_TRANSLATION_X)
    ty = random.uniform(-MAX_TRANSLATION_Y, MAX_TRANSLATION_Y)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return translated, angle + tx * STEERING_PER_PIXEL


def _random_flip(image, angle):
    if random.random() < 0.5:
        return cv2.flip(image, 1), -angle
    return image, angle


def _random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= random.uniform(0.7, 1.3)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _random_night(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= random.uniform(0.25, 0.6)
    hsv[:, :, 1] *= random.uniform(0.85, 1.15)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Add slight sensor noise common in low-light captures.
    noise = np.random.normal(0, 6, output.shape).astype(np.float32)
    output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return output


def _random_shadow(image):
    height, width = image.shape[:2]
    top_x = random.randint(0, width - 1)
    bottom_x = random.randint(0, width - 1)
    x_coords, y_coords = np.mgrid[0:height, 0:width]

    shadow_mask = (y_coords - top_x) * height - (bottom_x - top_x) * x_coords > 0
    if random.random() < 0.5:
        shadow_mask = np.logical_not(shadow_mask)

    shadow_ratio = random.uniform(0.4, 0.75)
    output = image.astype(np.float32)
    output[shadow_mask] *= shadow_ratio
    return np.clip(output, 0, 255).astype(np.uint8)


def _random_blur_or_noise(image):
    output = image
    if random.random() < 0.3:
        kernel = random.choice((3, 5))
        output = cv2.GaussianBlur(output, (kernel, kernel), 0)

    if random.random() < 0.3:
        noise = np.random.normal(0, 8, output.shape).astype(np.float32)
        output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return output


def _augment(image, angle):
    image, angle = _random_translate(image, angle)

    if random.random() < 0.25:
        image = _random_night(image)
    if random.random() < 0.8:
        image = _random_brightness(image)
    if random.random() < 0.5:
        image = _random_shadow(image)
    if random.random() < 0.4:
        image = _random_blur_or_noise(image)

    image, angle = _random_flip(image, angle)
    return image, angle


def _read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping image {image_path} because it could not be read")
    return image


def _pad_batch_if_needed(x_out, y_out, batch_size):
    if not x_out:
        raise RuntimeError("Could not build a batch because all sampled images were invalid")

    while len(x_out) < batch_size:
        copy_index = random.randrange(len(x_out))
        x_out.append(x_out[copy_index].copy())
        y_out.append(y_out[copy_index][:])


_load_dataset()
xs, ys = _sort_temporal(xs, ys)
xs, ys = _filter_missing_and_outliers(xs, ys)
ys = _smooth_steering_labels(ys)
xs, ys = _shuffle_pairs(xs, ys)

num_images = len(xs)
if num_images < 2:
    raise RuntimeError("Need at least 2 samples in data.txt to create train/validation splits")

split_index = max(1, min(num_images - 1, int(num_images * 0.8)))

train_xs = xs[:split_index]
train_ys = ys[:split_index]
val_xs = xs[split_index:]
val_ys = ys[split_index:]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

train_turn_indices = [idx for idx, angle in enumerate(train_ys) if abs(angle) >= TURN_ABS_THRESHOLD]


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    attempts = 0
    max_attempts = batch_size * 12

    while len(x_out) < batch_size and attempts < max_attempts:
        base_index = train_batch_pointer % num_train_images
        train_batch_pointer += 1

        if train_turn_indices and random.random() < TURN_SAMPLE_PROB:
            index = random.choice(train_turn_indices)
        else:
            index = base_index

        attempts += 1
        image_path = train_xs[index]
        image = _read_image(image_path)
        if image is None:
            continue

        angle = train_ys[index]
        image, angle = _augment(image, angle)
        image = _crop_resize_normalize(image)

        x_out.append(image)
        y_out.append([angle])

    _pad_batch_if_needed(x_out, y_out, batch_size)
    return x_out, y_out


def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    attempts = 0
    max_attempts = batch_size * 12

    while len(x_out) < batch_size and attempts < max_attempts:
        index = val_batch_pointer % num_val_images
        val_batch_pointer += 1
        attempts += 1

        image_path = val_xs[index]
        image = _read_image(image_path)
        if image is None:
            continue

        image = _crop_resize_normalize(image)
        x_out.append(image)
        y_out.append([val_ys[index]])

    _pad_batch_if_needed(x_out, y_out, batch_size)
    return x_out, y_out


# -----------------------------------------------------------------------------
# Legacy implementation preserved (commented) as requested by user.
# -----------------------------------------------------------------------------
# import cv2
# import os
# import random
# import numpy as np
#
# # Base directory resolution (robust path handling)
# BASE_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/driving_dataset"))
#
# xs=[]
# ys=[]
#
# train_batch_pointer=0
# val_batch_pointer=0
#
# with open(os.path.join(BASE_PATH,"data.txt")) as f:
#     for line in f:
#         img_file=line.split()[0]
#         angle_str=line.split()[1].split(',')[0]
#         xs.append(os.path.join(BASE_PATH, img_file))
#         #the paper by Nvidia uses the inveses of the turning radius,
#         #but steering wheel angle is proportional to the inverse of turning radius
#         #so the steering wheel angle in radians is used as outputs(pi/180)
#         ys.append(float(angle_str) * np.pi /180)
#
# #get number of images
# num_images=len(xs)
# #shuffle list of images
# c=list(zip(xs, ys))
# # random.shuffle(c)
# xs,ys=zip(*c)
#
# train_xs=xs[:int(len(xs)*0.8)]
# train_ys=ys[:int(len(xs)*0.8)]
#
# val_xs=xs[-int(len(xs)*0.2):]
# val_ys=ys[-int(len(xs)*0.2):]
#
# num_train_images=len(train_xs)
# num_val_images=len(val_xs)
#
#
#
# def LoadTrainBatch(batch_size):
#     global train_batch_pointer
#     x_out=[]
#     y_out=[]
#
#     for i in range(batch_size):
#         idx=(train_batch_pointer+i)%num_train_images
#         image_path=train_xs[idx]
#         img=cv2.imread(image_path)
#         if img is None:
#             print(f"Skipping image {image_path} as it is missing")
#             continue
#         img=img[-150:]
#         img=cv2.resize(img,(200,66))/255.0
#         x_out.append(img)
#         y_out.append([train_ys[idx]])
#     train_batch_pointer+=batch_size
#     return x_out, y_out
#
#
# def LoadValBatch(batch_size):
#     global val_batch_pointer
#     x_out = []
#     y_out = []
#     for i in range(batch_size):
#         image_path = val_xs[(val_batch_pointer + i) % num_val_images]
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"Skipping image {image_path} as it is missing")
#             continue  # we will skip this image
#         img = img[-150:]  # crop the image(90 pixels from top)
#         img = cv2.resize(img, (200, 66)) / 255.0
#         x_out.append(img)
#         y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
#         #same hi krna hai ismein bhi
#     val_batch_pointer += batch_size
#     return x_out, y_out
