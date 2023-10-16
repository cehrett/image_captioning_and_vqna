# Import libraries
import os

def prepare_to_load_model(username, service='huggingface', use_personal=False):
    """
    Set cache directory and load api key for Huggingface
    username should be your clemson username. This will be used to set the location of your scratch directory. 
    We need to store models in the scratch directory because the models are too big for the home directory.
    """

    directory_path = os.path.join('/scratch',username)

    # Set Huggingface cache directory to be on scratch drive
    if os.path.exists(directory_path):
        hf_cache_dir = os.path.join(directory_path,'hf_cache')
        if not os.path.exists(hf_cache_dir):
            os.mkdir(hf_cache_dir)
        print(f"Okay, using {hf_cache_dir} for huggingface cache. Models will be stored there.")
        assert os.path.exists(hf_cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = f'/scratch/{username}/hf_cache/'
    else:
        error_message = f"Are you sure you entered your username correctly? I couldn't find a directory {directory_path}."
        raise FileNotFoundError(error_message)

    # Load Huggingface api key
    if service.lower() == 'huggingface':
        api_key_loc = os.path.join('/home', username, '.apikeys', 'huggingface_api_key.txt')
        # Use text file at api_key_loc and os library to export environment variable HUGGINGFACE_APIKEY
        with open(api_key_loc, 'r') as file:
            huggingface_api_key = file.read().replace('\n', '')
        os.environ["HUGGINGFACE_APIKEY"] = huggingface_api_key
    elif service.lower() == 'openai':
        raise NotImplementedError("No OpenAI models are implemented here for image generation.")        
    if os.path.exists(api_key_loc):
        print(f'{service} API key loaded.')
    else:
        error_message = f'{service} API key not found.'
        raise FileNotFoundError(error_message)