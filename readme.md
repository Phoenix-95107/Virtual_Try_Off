# TEMU-VTOFF: Virtual Try-Off & Fashion Understanding Toolkit
TEMU-VTOFF is a state-of-the-art toolkit for virtual try-off and fashion image understanding. It leverages advanced diffusion models, vision-language models, and semantic segmentation to enable garment transfer, attribute captioning, and mask generation for fashion images.
`<img src="./assets/teaser.png" alt="example">`
## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
  - [1. Inference Pipeline (`inference.py`)](#1-inference-pipeline-inferencepy)
  - [2. Visual Attribute Captioning (`precompute_utils/captioning_qwen.py`)](#2-visual-attribute-captioning-precompute_utilscaptioning_qwenpy)
  - [3. Clothing Segmentation (`SegCloth.py`)](#3-clothing-segmentation-segclothpy)
- [Examples](#examples)
- [Citation](#citation)
- [License](#license)

---

## Features

- **Virtual Try-On**: Generate realistic try-on images using Stable Diffusion 3-based pipelines.
- **Visual Attribute Captioning**: Extract fine-grained garment attributes using Qwen-VL.
- **Clothing Segmentation**: Obtain binary and fine masks for garments using SegFormer.
- **Dataset Support**: Works with DressCode and VITON-HD datasets.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/TEMU-VTOFF.git
   cd TEMU-VTOFF
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Setup virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

---

## Quick Start

### 1. Virtual Try-On Inference

```bash
python inference.py \
  --pretrained_model_name_or_path <path/to/model> \
  --pretrained_model_name_or_path_sd3_tryoff <path/to/tryoff/model> \
  --example_image examples/example1.jpg \
  --output_dir outputs \
  --width 768 --height 1024 \
  --guidance_scale 2.0 \
  --num_inference_steps 28 \
  --category upper_body
```

### 2. Visual Attribute Captioning

```bash
python precompute_utils/captioning_qwen.py \
  --pretrained_model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --image_path examples/example1.jpg \
  --output_path outputs/example1_caption.txt \
  --image_category upper_body
```

### 3. Clothing Segmentation

```python
from PIL import Image
from SegCloth import segment_clothing

img = Image.open("examples/example1.jpg")
binary_mask, fine_mask = segment_clothing(img, category="upper_body")
binary_mask.save("outputs/example1_binary_mask.jpg")
fine_mask.save("outputs/example1_fine_mask.jpg")
```

---

## Core Components

### 1. Inference Pipeline (`inference.py`)

- **Purpose**: Generates virtual try-on images using a Stable Diffusion 3-based pipeline.
- **How it works**:
  - Loads pretrained models (VAE, transformers, schedulers, encoders).
  - Segments the clothing region using `SegCloth.py`.
  - Generates a descriptive caption for the garment using Qwen-VL (`captioning_qwen.py`).
  - Runs the diffusion pipeline to synthesize a new try-on image.
- **Key Arguments**:
  - `--pretrained_model_name_or_path`: Path or HuggingFace model ID for the main model.
  - `--pretrained_model_name_or_path_sd3_tryoff`: Path or ID for the try-off transformer.
  - `--example_image`: Input image path.
  - `--output_dir`: Output directory.
  - `--category`: Clothing category (`upper_body`, `lower_body`, `dresses`).
  - `--width`, `--height`: Output image size.
  - `--guidance_scale`, `--num_inference_steps`: Generation parameters.

### 2. Visual Attribute Captioning (`precompute_utils/captioning_qwen.py`)

- **Purpose**: Generates fine-grained, structured captions for fashion images using Qwen2.5-VL.
- **How it works**:
  - Loads the Qwen2.5-VL model and processor.
  - For a given image, predicts garment attributes (e.g., type, fit, hem, neckline) in a controlled, structured format.
  - Can process single images or entire datasets (DressCode, VITON-HD).
- **Key Arguments**:
  - `--pretrained_model_name_or_path`: Path or HuggingFace model ID for Qwen2.5-VL.
  - `--image_path`: Path to a single image (for single-image captioning).
  - `--output_path`: Where to save the generated caption.
  - `--image_category`: Garment category (`upper_body`, `lower_body`, `dresses`).
  - For batch/dataset mode: `--dataset_name`, `--dataset_root`, `--filename`.

### 3. Clothing Segmentation (`SegCloth.py`)

- **Purpose**: Segments clothing regions in images, producing:
  - A binary mask (black & white) of the garment.
  - A fine mask image where the garment is grayed out.
- **How it works**:
  - Uses a SegFormer model (`mattmdjaga/segformer_b2_clothes`) via HuggingFace `transformers` pipeline.
  - Supports categories: `upper_body`, `dresses`, `lower_body`.
  - Provides both single-image and batch processing functions.
- **Usage**:
  - `segment_clothing(img, category)`: Returns `(binary_mask, fine_mask)` for a PIL image.
  - `batch_segment_clothing(img_dir, out_dir)`: Processes all images in a directory.

---

## Examples

See the `examples/` directory for sample images, masks and captions. Example usage scripts are provided for each core component.
Here is the workflow of this model and a comparison of its results with other models.
**Workflow
<img src="./assets/workflow.png" alt="Workflow" />
**Compair
<img src="./assets/compair.png" alt="compair" />
---

## Citation

If you use TEMU-VTOFF in your research or product, please cite this repository and the relevant models (e.g., Stable Diffusion 3, Qwen2.5-VL, SegFormer).

```
@misc{temu-vtoff,
  author = {Your Name or Organization},
  title = {TEMU-VTOFF: Virtual Try-On & Fashion Understanding Toolkit},
  year = {2024},
  howpublished = {\url{https://github.com/yourusername/TEMU-VTOFF}}
}
```

---

## License

This project is licensed under the [LICENSE](LICENSE) provided in the repository. Please check individual model and dataset licenses for additional terms.
