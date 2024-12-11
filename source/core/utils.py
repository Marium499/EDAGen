import pandas as pd
import re
from PIL import Image

import base64
from io import BytesIO

def random_sample_dataframe(df_path, n, random_state=None):
    """
    Randomly samples n rows from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to sample from.
    - n (int): The number of rows to sample.
    - random_state (int, optional): A seed for reproducibility. Default is None.

    Returns:
    - pd.DataFrame: A DataFrame with n randomly sampled rows.
    """
    df = pd.read_csv(df_path, header=0)
    if n > len(df):
        raise ValueError("n cannot be greater than the number of rows in the DataFrame.")
    return df.sample(n=n, random_state=random_state)


def extract_bullets(input, pat = "numbered_bullet_pattern"):

    # parse response to create list
    pattern_dict = {
        "bullet_pattern" : r"^- (.+)$",
        "numbered_bullet_pattern" : r"^\d+\.\s(.+)$",
        "item_pattern" : r"item:\s*(.+)",
    }

    # Parse the text and extract the bullet points
    b_list = re.findall(pattern_dict[pat], input, re.MULTILINE)
    return b_list


def process_outputs(input_string, img_ext=".png"):
    pattern = r"Read chart from path (\S+)"
    
    # Find all matches
    matches = re.findall(pattern, input_string)
    if matches:
        image_results = []
        for image_path in matches:
            print(f"Extracted Filename: {image_path}")
            try:
                image = Image.open(image_path)
                print("Image loaded successfully!")
                img_base64 = pil_to_base64(image)
                image_results.append({
                    "image_path": image_path,
                    # "image_output": image,
                    "image_base64": img_base64
                })
            except FileNotFoundError:
                print(f"Image not found. Please check the file path: {image_path}")
                image_results.append({
                    "image_path": image_path,
                    # "image_output": None,
                    "image_base64": None
                })
        
        # Remove matched patterns from input_string
        modified_text = re.sub(pattern, "", input_string).strip()
        #should probably raise exception here because image should exist in file path #####
        return {"text_output": modified_text, "images":image_results}
    else:
        print("No filenames found.")
        return {"text_output": input_string, "images": None}


# Convert PIL image to Base64 string
def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# # Usage
# base64_image = pil_to_base64(pil_image)
# prompt = f"This is the Base64 representation of an image: {base64_image}. What should I do with it?"

