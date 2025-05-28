import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
import cv2
import argparse

hf_dir = snapshot_download(repo_id="alexanderkroner/MSI-Net")
model = tf.keras.models.load_model(hf_dir)


def get_target_shape(original_shape):
    original_aspect_ratio = original_shape[0] / original_shape[1]

    square_mode = abs(original_aspect_ratio - 1.0)
    landscape_mode = abs(original_aspect_ratio - 240 / 320)
    portrait_mode = abs(original_aspect_ratio - 320 / 240)

    best_mode = min(square_mode, landscape_mode, portrait_mode)

    if best_mode == square_mode:
        target_shape = (320, 320)
    elif best_mode == landscape_mode:
        target_shape = (240, 320)
    else:
        target_shape = (320, 240)

    return target_shape


def preprocess_input(input_image, target_shape):
    input_tensor = tf.expand_dims(input_image, axis=0)

    input_tensor = tf.image.resize(
        input_tensor, target_shape, preserve_aspect_ratio=True
    )

    vertical_padding = target_shape[0] - input_tensor.shape[1]
    horizontal_padding = target_shape[1] - input_tensor.shape[2]

    vertical_padding_1 = vertical_padding // 2
    vertical_padding_2 = vertical_padding - vertical_padding_1

    horizontal_padding_1 = horizontal_padding // 2
    horizontal_padding_2 = horizontal_padding - horizontal_padding_1

    input_tensor = tf.pad(
        input_tensor,
        [
            [0, 0],
            [vertical_padding_1, vertical_padding_2],
            [horizontal_padding_1, horizontal_padding_2],
            [0, 0],
        ],
    )

    return (
        input_tensor,
        [vertical_padding_1, vertical_padding_2],
        [horizontal_padding_1, horizontal_padding_2],
    )


def postprocess_output(
    output_tensor, vertical_padding, horizontal_padding, original_shape
):
    output_tensor = output_tensor[
        :,
        vertical_padding[0] : output_tensor.shape[1] - vertical_padding[1],
        horizontal_padding[0] : output_tensor.shape[2] - horizontal_padding[1],
        :,
    ]

    output_tensor = tf.image.resize(output_tensor, original_shape)

    output_array = output_tensor.numpy().squeeze()
    output_array = plt.cm.inferno(output_array)[..., :3]

    return output_array


def test_saliency(image_path):
    input_image = tf.keras.utils.load_img(image_path)
    input_image = np.array(input_image, dtype=np.float32)

    original_shape = input_image.shape[:2]
    target_shape = get_target_shape(original_shape)

    input_tensor, vertical_padding, horizontal_padding = preprocess_input(
        input_image, target_shape
    )


    output_tensor = model(input_tensor)["output"]


    saliency_map = postprocess_output(
        output_tensor, vertical_padding, horizontal_padding, original_shape
    )

    alpha = 0.65

    blended_image = alpha * saliency_map + (1 - alpha) * input_image / 255

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image / 255)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(blended_image)
    plt.title("Saliency Map")
    plt.axis("off")

    plt.tight_layout()
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close('all')
    return
    

def apply_msi_blended(file_path, out_path):
    if os.path.exists(file_path):
        print(f'apply_msi_blended(): processing {file_path}')
        input_image = tf.keras.utils.load_img(file_path)
        input_image = np.array(input_image, dtype=np.float32)

        original_shape = input_image.shape[:2]
        target_shape = get_target_shape(original_shape)

        input_tensor, vertical_padding, horizontal_padding = preprocess_input(
            input_image, target_shape
        )
        output_tensor = model(input_tensor)["output"]
        saliency_map = postprocess_output(
            output_tensor, vertical_padding, horizontal_padding, original_shape
        )
        alpha = 0.65
        blended_image = alpha * saliency_map + (1 - alpha) * input_image / 255
        blended_image = (blended_image * 255).astype(np.uint8)
        blended_image = cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, blended_image)
    else:
        print(f'apply_msi_blended(): file {file_path} does not exist')
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ops', type=str, required=False, default='apply_msi_blended')
    parser.add_argument('--file_path', type=str, required=False)
    parser.add_argument('--out_path', type=str, required=False)
    args = parser.parse_args()
    if args.ops == 'apply_msi_blended':
        apply_msi_blended(args.file_path, args.out_path)
    else:
        print(f'Invalid operation: {args.ops}')