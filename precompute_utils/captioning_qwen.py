from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, pipeline
# from qwen_vl_utils import process_vision_info
import os
from accelerate.utils import set_seed
import torch
import json
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from tqdm import tqdm

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to the pretrained Qwen VL model.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        choices=["dresscode", "vitonhd"],
        help="Name of the dataset to process (dresscode or vitonhd).",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        required=False,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        required=False,
        help=
        "Path to visual attributes annotations for RAG enhanced VLM captioning",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help=
        "Path to a single image to caption. If provided, dataset arguments are ignored."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=
        "Path to save the generated caption. If not provided, it will be saved next to the image with a .txt extension."
    )
    parser.add_argument("--image_category",
                        type=str,
                        default="upper_body",
                        choices=["dresses", "upper_body", "lower_body"],
                        help="Category of the cloth in the image.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.2])

    args = parser.parse_args(input_args)

    if args.image_path is None:
        if args.dataset_name is None or args.dataset_root is None or args.filename is None:
            parser.error(
                "When not using --image_path, you must specify --dataset_name, --dataset_root, and --filename."
            )

    return args


def caption_single_image(img_path, category, model=model, processor=processor):
    visual_attributes = {
        "dresses": [
            "Cloth Type", "Waist", "Fit", "Hem", "Neckline", "Sleeve Length",
            "Cloth Length"
        ],
        "upper_body": [
            "Cloth Type", "Waist", "Fit", "Hem", "Neckline", "Sleeve Length",
            "Cloth Length"
        ],
        "lower_body": ["Cloth Type", "Waist", "Fit", "Cloth Length"]
    }

    if not os.path.exists(img_path):
        print(f"Error: Image path not found at {img_path}")
        return

    messages = [{
        "role":
        "system",
        "content":
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    }, {
        "role":
        "user",
        "content": [
            {
                "type": "image",
                "image": f"{img_path}",
            },
            {
                "type":
                "text",
                "text":
                "Use only visual attributes that are present in the image. "
                f"Predict values of the following attributes: {visual_attributes[category]}. "
                "It's forbidden to generate the following visual attributes: colors, background and textures/patterns. "
                "It's forbidden to generate unspecified predictions. It's forbidden to generate newlines characters. "
                f"Generate in this way: a <cloth type> with <attributes description>"
            },
        ],
    }]

    text = processor.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    temperature = 0.2

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=temperature,
        do_sample=True,
        pad_token_id=processor.tokenizer.eos_token_id)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

    caption = output_text[0]
    print(f"Generated Caption: {caption}")

    output_path = args.output_path
    if output_path is None:
        base, _ = os.path.splitext(args.image_path)
        output_path = f"{base}.txt"

    with open(output_path, "w") as f:
        f.write(caption)

    print(f"Caption saved to: {output_path}")


def qwen_captioning(args):
    set_seed(42)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        use_fast=True,
        trust_remote_code=True)

    if args.image_path:
        caption_single_image(args, model, processor)
        return

    visual_attributes = {
        "dresses": [
            "Cloth Type", "Waist", "Fit", "Hem", "Neckline", "Sleeve Length",
            "Cloth Length"
        ],
        "upper_body": [
            "Cloth Type", "Waist", "Fit", "Hem", "Neckline", "Sleeve Length",
            "Cloth Length"
        ],
        "lower_body": ["Cloth Type", "Waist", "Fit", "Cloth Length"]
    }

    outputlist = ["c_name", "category"]
    category = ["upper_body", "lower_body", "dresses"]
    if args.dataset_name == "dresscode":
        train_dataset = DressCodeDataset(
            dataroot_path=args.dataset_root,
            phase="train",
            order="paired",
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            category=category,
            size=(args.height, args.width),
        )
        test_dataset = DressCodeDataset(
            dataroot_path=args.dataset_root,
            phase="test",
            order="paired",
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            category=category,
            size=(args.height, args.width),
        )
    elif args.dataset_name == "vitonhd":
        train_dataset = VitonHDDataset(
            dataroot_path=args.dataset_root,
            phase="train",
            order="paired",
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            size=(args.height, args.width),
        )
        test_dataset = VitonHDDataset(
            dataroot_path=args.dataset_root,
            phase="test",
            order="paired",
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            size=(args.height, args.width),
        )

    dataroot_names = train_dataset.dataroot_names + test_dataset.dataroot_names
    phases = ["train" for _ in range(len(train_dataset))
              ] + ["test" for _ in range(len(test_dataset))]
    c_names = train_dataset.c_names + test_dataset.c_names
    temperatures = args.temperatures
    vlm_captions_dicts = [{} for _ in range(len(temperatures))]

    for i, t in enumerate(temperatures):
        t = str(temperatures[i]).replace(".", "_")
        if os.path.exists(
                os.path.join(args.dataset_root,
                             args.filename.replace(".json", f"_{t}.json"))):
            with open(
                    os.path.join(args.dataset_root,
                                 args.filename.replace(".json",
                                                       f"_{t}.json"))) as f:
                vlm_captions_dicts[i] = json.load(f)

    idx = 0
    for dataroot, phase, c_name in tqdm(zip(dataroot_names, phases, c_names)):
        idx += 1
        item_id = c_name.split("_")[0]
        if item_id in vlm_captions_dicts[0].keys():
            continue

        if args.dataset_name == "vitonhd":
            category = "upper_body"
            img_path = os.path.join(dataroot, phase, 'cloth', c_name)

        elif args.dataset_name == "dresscode":
            category = dataroot.split("/")[-1]
            img_path = os.path.join(dataroot, 'cleaned_inshop_imgs',
                                    c_name.replace(".jpg", "_cleaned.jpg"))
            if not os.path.exists(img_path):
                img_path = os.path.join(dataroot, 'images', c_name)

        messages = [{
            "role":
            "system",
            "content":
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        }, {
            "role":
            "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{img_path}",
                },
                {
                    "type":
                    "text",
                    "text":
                    "Use only visual attributes that are present in the image. "
                    f"Predict values of the following attributes: {visual_attributes[category]}. "
                    "It's forbidden to generate the following visual attributes: colors, background and textures/patterns. "
                    "It's forbidden to generate unspecified predictions. It's forbidden to generate newlines characters. "
                    f"Generate in this way: a <cloth type> with <attributes description>"
                },
            ],
        }]

        # Preparation for inference
        text = processor.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        for i, temperature in enumerate(temperatures):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=75,
                temperature=temperature,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            vlm_captions_dicts[i][item_id] = output_text

        if idx % 1000 == 0:
            for i, vlm_captions_dict in enumerate(vlm_captions_dicts):
                t = str(temperatures[i]).replace(".", "_")
                with open(
                        os.path.join(
                            args.dataset_root,
                            args.filename.replace(".json", f"_{t}.json")),
                        "w") as f:
                    json.dump(vlm_captions_dict, f)

    for i, vlm_captions_dict in enumerate(vlm_captions_dicts):
        t = str(temperatures[i]).replace(".", "_")
        with open(
                os.path.join(args.dataset_root,
                             args.filename.replace(".json", f"_{t}.json")),
                "w") as f:
            json.dump(vlm_captions_dict, f)


if __name__ == "__main__":
    args = parse_args()
    qwen_captioning(args)
