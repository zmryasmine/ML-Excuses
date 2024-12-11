import numpy as np
from PIL import Image
from skimage import io, transform, color

def process_image(filepath):
    """
    Convert the uploaded image to a format suitable for the MLflow API.
    This function performs the same transformations as described in the second code snippet.
    """
    # Load the image using skimage for consistency with the second snippet
    image = io.imread(filepath)

    # Resize the image to 100x100 with 3 channels
    image = transform.resize(image, (100, 100, 3), anti_aliasing=True)

    # Convert the image to grayscale
    image = color.rgb2gray(image)

    # Flatten the grayscale image into a 1D vector
    flattened = image.flatten().tolist()

    # Create the input dictionary
    input_data = {
        "dataframe_records": [{"pixel_" + str(i): value for i, value in enumerate(flattened)}]
    }

    return input_data
