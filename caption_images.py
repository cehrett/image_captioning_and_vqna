# Batch image captioning script
# 2023-10-26 C Ehrett

from utils.prepare_apikey_and_cache_loc import prepare_to_load_model 
import os
username = os.environ.get('USER')
prepare_to_load_model(username=username)

# Import needed libraries
from PIL import Image
from IPython.display import display
from IPython.display import Image as IPImage
import torch
import time
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering, AutoTokenizer
import argparse
import logging
import json


# Define the caption prompt
CAPTION_QUESTION = """
Question: Please provide a comprehensively detailed description of the image. \

Answer: The image shows \
"""

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def describe_batch_images(model, 
    tokenizer, 
    processor, 
    image_locs,
    device, 
    questions=CAPTION_QUESTION,  
    include_question=True,
    temperature=.15,
    top_p=0.95):
    """
    Generates descriptions for the given batch of images using the provided model and tokenizer.

    Args:
    - model (torch.nn.Module): The pretrained model for image description.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
    - processor (object): The processor for the images to convert them to pixel values.
    - image_locs (List[str]): The list of input image file locations.
    - device (str, optional): Device to move tensors to. Default is 'cuda:0'. (model should already be there.)
    - questions (Union[str, List[str]]): The prompt or list of prompts for the model.
    - temperature (float): Text generation temperature.
    - top_p (float): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.

    Returns:
    - List[str]: Decoded descriptions of the images.
    """
    
    # Check if a single question is provided, if so, repeat it for each image
    if isinstance(questions, str):
        questions = [questions] * len(image_locs)
    
    # Ensure the length of questions and image_locs is the same
    assert len(questions) == len(image_locs), "Number of questions and image locations should be the same."
    
    # Preprocess each image and store pixel values
    pixel_values_list = []

    for loc in image_locs:
        image = Image.open(loc)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        pixel_values_list.append(torch.from_numpy(processor(image).pixel_values[0]))

    # Stack all pixel values into a batch tensor
    pixel_values = torch.stack(pixel_values_list).to(device)

    # Prepare generation kwargs
    gen_kwargs = {'temperature':temperature, 'do_sample':True}
    if include_question:
        # Tokenize the questions and move tensors to the specified device
        encoding = tokenizer(questions, return_tensors='pt', padding=True, truncation=True)
        gen_kwargs['input_ids'] = encoding['input_ids'].to(device)
        gen_kwargs['attention_mask'] = encoding['attention_mask'].to(device)

    # Time the generation process
    start_time = time.time()

    output = model.generate(
        pixel_values=pixel_values,
        max_new_tokens=200,
        **gen_kwargs
    )

    end_time = time.time()

    # Decode each output tensor
    decoded_outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in output]

    # Print the elapsed time
    print(f"Time taken to generate output: {end_time - start_time:.2f} seconds")

    return decoded_outputs


def main(args):
    # Your main execution logic here, for example:
    # model = LoadYourModelHere()
    # tokenizer = LoadYourTokenizerHere()
    # processor = LoadYourProcessorHere()
    
    model_name = args.model_name

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        logging.warning("CUDA is not available. Running on CPU.")

    start_time = time.time()
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForVisualQuestionAnswering.from_pretrained(model_name).to(DEVICE)
    end_time = time.time()
    # Print the elapsed time
    print(f"Time taken to load model: {end_time - start_time:.2f} seconds")
    
    descriptions = describe_batch_images(model, 
                                         tokenizer, 
                                         processor, 
                                         args.image_locs, 
                                         DEVICE, 
                                         temperature=args.temperature, 
                                         top_p=args.top_p)
    
    # Create a dictionary of image locations and descriptions
    image_description_map = dict(zip(args.image_locs, descriptions))

    # Serialize and save to the specified JSON file
    with open(args.output_json, 'w') as json_file:
        json.dump(image_description_map, json_file, indent=4)

    # Log the descriptions:
    for loc, desc in image_description_map.items():
        logger.info(f"{loc}: {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image captioning script.')
    parser.add_argument('--image_locs', nargs='+', required=True, help='List of image file locations')
    parser.add_argument('--temperature', type=float, default=1, help='Text generation temperature.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Probability threshold for token generation.')
    parser.add_argument('--model_name', type=str, default="Salesforce/blip2-flan-t5-xl", help='Model name to be loaded from HuggingFace Model Hub.')
    parser.add_argument('--output_json', type=str, default='image_descriptions.json', help='Path to the output JSON file with descriptions.')
    # Add other arguments as needed, for example, model_path, etc.
    args = parser.parse_args()
    
    main(args)


