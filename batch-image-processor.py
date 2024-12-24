import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel
)
from pathlib import Path
import time
from tqdm import tqdm
import json


def setup_models():
    # Caption model
    caption_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base").to("cuda")

    # Tag model
    tag_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32")
    tag_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32").to("cuda")

    return caption_processor, caption_model, tag_processor, tag_model


def get_tags(image, processor, model, candidate_labels):
    inputs = processor(images=image, text=candidate_labels,
                       return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    return [label for prob, label in zip(probs[0], candidate_labels) if prob > 0.3]


def process_batch(image_paths, processors, models, batch_size=4):
    caption_processor, caption_model, tag_processor, tag_model = models
    results = []

    candidate_labels = [
        "people", "indoor", "outdoor", "day", "night", "work", "meeting",
        "computer", "desk", "office", "casual", "formal", "group", "solo"
    ]

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        caption_inputs = []
        images = []

        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                caption_input = caption_processor(
                    image, return_tensors="pt").to("cuda")
                caption_inputs.append(caption_input)
                images.append(image)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # Generate captions
        with torch.no_grad():
            batch_inputs = {
                "pixel_values": torch.cat([inp["pixel_values"] for inp in caption_inputs])
            }
            generated_ids = caption_model.generate(
                **batch_inputs, max_length=50)
            captions = caption_processor.batch_decode(
                generated_ids, skip_special_tokens=True)

            # Generate tags
            for img, path, caption in zip(images, batch_paths, captions):
                tags = get_tags(img, tag_processor,
                                tag_model, candidate_labels)
                results.append({
                    "image_path": str(path),
                    "caption": caption,
                    "tags": tags
                })

    return results


def main():
    image_dir = Path("./images")
    output_file = "results.json"
    batch_size = 4

    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    processors_and_models = setup_models()

    start_time = time.time()
    results = process_batch(
        image_paths, processors_and_models, batch_size=batch_size)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Processed {len(results)} images in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
