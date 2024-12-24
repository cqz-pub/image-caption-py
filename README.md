# Batch Image Processor

This project is a batch image processor that generates captions and tags for images using state-of-the-art models. It leverages the power of deep learning to analyze images and provide a descriptive caption and relevant tags.

## Features

- **Batch Processing**: Process multiple images at once, making it efficient for large datasets.
- **Image Captioning**: Generates a descriptive caption for each image using the BLIP model.
- **Image Tagging**: Identifies and tags images with relevant keywords using the CLIP model.
- **Customizable**: Supports custom batch sizes and candidate labels for tagging.

## Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.1.0 or higher
- **Pillow**: 9.0.0 or higher
- **tqdm**: 4.65.0 or higher
- **pathlib**: Latest version

## Usage

1. Clone the repository and navigate to the project directory.
2. Install the required packages by running `pip install -r requirements.txt`.
3. Place your images in the `images` directory.
4. Run the script using `python batch-image-processor.py`.
5. The results will be saved in a JSON file named `results.json`.

## Configuration

You can customize the batch size and candidate labels for tagging by modifying the `process_batch` function in `batch-image-processor.py`.

## Output

The output will be a JSON file containing the image path, caption, and tags for each processed image.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

This project uses the following open-source models:
- [BLIP](https://github.com/salesforce/BLIP) by Salesforce
- [CLIP](https://github.com/openai/CLIP) by OpenAI

