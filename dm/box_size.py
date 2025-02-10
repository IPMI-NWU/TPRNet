#"/home/gaozhizezhang/jzproject/data/DeepGlobe/gt/"
import os
import json
import torch
from PIL import Image
import numpy as np
import os,sys
import math
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, parent_dir)
from utils.fore_extraction import getForeCoord

def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert('L')
    image = np.array(image)
    image = torch.from_numpy(image).float()
    image = image / 255.0 
    image = image.unsqueeze(0).unsqueeze(0)  
    return image

def save_coordinates_to_file(coordinates, output_file="coordinates.json"):

    with open(output_file, 'w') as f:
        json.dump(coordinates, f, indent=4)

def process_images_in_directory(image_dir, output_file="coordinates.json"):

    coordinates = {}
    
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):

            image_path = os.path.join(image_dir, filename)

            image_tensor = load_image_as_tensor(image_path)

            start_coord, end_coord = getForeCoord(image_tensor)
            height=int(end_coord[1]-start_coord[1])
            width=int(end_coord[2]-start_coord[2])
            area=height*width
            length=int(math.sqrt(area))
            length=length+1 if length%2==1 else length
            coordinates[filename] = {"height": height, "width": width,"area":area,"length":length}
    

    save_coordinates_to_file(coordinates, output_file)


if __name__ == "__main__":

    image_directory = "/home/gaozhizezhang/jzproject/data/DeepGlobe/gt/" 
    output_file = 'coordinates.json'
    process_images_in_directory(image_directory, output_file)
