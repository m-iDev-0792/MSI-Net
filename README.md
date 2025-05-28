---
library_name: tf-keras
license: mit
pipeline_tag: image-to-image
---

MSI-Net 
=======

📖 [Contextual encoder-decoder network for visual saliency prediction](https://doi.org/10.1016/j.neunet.2020.05.004)

🤗 A demo of this model can be found on [HuggingFace Spaces](https://huggingface.co/spaces/alexanderkroner/saliency).

---

<img src="https://github.com/alexanderkroner/saliency/blob/master/figures/results.jpg?raw=true" width="800">

# Summary

MSI-Net is a visual saliency model that predicts where humans fixate on natural images using a contextual encoder-decoder network trained on eye movement data. The model is based on a convolutional neural network architecture and includes an ASPP module with multiple convolutional layers at different dilation rates to capture multi-scale features in parallel. Moreover, it combines the resulting representations with global scene information towards accurate predictions of visual saliency. MSI-Net consists of roughly 25M parameters and thus presents a suitable choice for applications with limited computational resources. For more information on the model, check out [GitHub](https://github.com/alexanderkroner/saliency) and the corresponding [paper](https://doi.org/10.1016/j.neunet.2020.05.004) or [preprint](https://arxiv.org/abs/1902.06634).

<img src="https://github.com/alexanderkroner/saliency/blob/master/figures/architecture.jpg?raw=true" width="700">

# Requirements

To install the required dependencies, use either `pip` or `conda`:
```
pip install -r requirements.txt
```
```
conda env create -f requirements.yml
```

# Example Use

### Import the dependencies
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
```

### Download the repo files
```python
hf_dir = snapshot_download(repo_id="alexanderkroner/MSI-Net")
```

### Load the saliency model
```python
model = tf.keras.models.load_model(hf_dir)
```

### Load the functions for preprocessing the input and postprocessing the output
```python
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
```

### Load and preprocess an example image
```python
input_image = tf.keras.utils.load_img(hf_dir + "/example.jpg")
input_image = np.array(input_image, dtype=np.float32)

original_shape = input_image.shape[:2]
target_shape = get_target_shape(original_shape)

input_tensor, vertical_padding, horizontal_padding = preprocess_input(
    input_image, target_shape
)
```

### Feed the input tensor to the model
```python
output_tensor = model(input_tensor)["output"]
```

### Postprocess and visualize the output
```python
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
plt.show()
```

# Datasets

Before training the model on fixation data, the encoder weights were initialized from a VGG16 backbone pre-trained on the ImageNet classification task. The model was then trained on the SALICON dataset, which consists of mouse movement recordings as a proxy for gaze measurements. Finally, the weights can be fine-tuned on human eye tracking data. MSI-Net was therefore also trained on one of the following datasets, although here we only provide the SALICON base model:

|             | Number of Images | Viewers per Image | Viewing Duration | Recording Type |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| [SALICON](https://doi.org/10.1109/CVPR.2015.7298710) | 10,000 | 16 | 5s | Mouse tracking |
| [MIT1003](https://doi.org/10.1109/ICCV.2009.5459462) | 1,003 | 15 | 3s | Eye tracking |
| [CAT2000](https://doi.org/10.48550/arXiv.1505.03581) | 4,000 | 24 | 5s | Eye tracking |
| [DUT-OMRON](https://doi.org/10.1109/CVPR.2013.407) | 5,168 | 5 | 2s | Eye tracking |
| [PASCAL-S](https://doi.org/10.1109/CVPR.2014.43) | 850 | 8 | 2s | Eye tracking |
| [OSIE](https://doi.org/10.1167/14.1.28) | 700 | 15 | 3s | Eye tracking |
| [FIWI](https://doi.org/10.1007/978-3-319-10584-0_3) | 149 | 11 | 5s | Eye tracking |

<br> Evaluations of our model are available on the original [MIT saliency benchmark](http://saliency.mit.edu/results.html) and the updated [MIT/Tübingen saliency benchmark](https://saliency.tuebingen.ai/results.html). Results for the latter are derived from a probabilistic representation of predicted saliency maps with metric-specific postprocessing for a fair model comparison.

# Limitations

MSI-Net was trained on human fixation data collected under a free-viewing paradigm. Therefore, the predicted saliency maps may not generalize to viewers that received task instructions during the experiment. It must also be noted that the training data consisted primarily of natural images. As a result, gaze predictions for specific image types (e.g., fractals, patterns) or adversarial examples may not be very accurate.

Another limitation is that saliency-based cropping algorithms, formerly applied to images uploaded to the social media platform Twitter between 2018 and 2021, have shown [biases in terms of race and gender](https://doi.org/10.1145/3479594). It is thus important to use saliency models with caution and acknowledge the potential risks that are involved in their application.

# Reference

If you find this code or model useful, please cite the following paper:

```
@article{kroner2020contextual,
  title={Contextual encoder-decoder network for visual saliency prediction},
  author={Kroner, Alexander and Senden, Mario and Driessens, Kurt and Goebel, Rainer},
  url={http://www.sciencedirect.com/science/article/pii/S0893608020301660},
  doi={https://doi.org/10.1016/j.neunet.2020.05.004},
  journal={Neural Networks},
  publisher={Elsevier},
  year={2020},
  volume={129},
  pages={261--270},
  issn={0893-6080}
}
```