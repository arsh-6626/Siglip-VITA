import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from model import tokenizer, tokenizerConfig
from PIL import Image
import torch 
import numpy as np
from transformer import VisionTransformer
import cv2
# from torchsummary import summary

class NewModel_2(torch.nn.Module):  # Inherit from torch.nn.Module
    def __init__(self):
        super(NewModel_2, self).__init__()  # Call the constructor of torch.nn.Module
        config = tokenizerConfig()
        self.model_tokenizer = tokenizer(config).to('cuda')
        self.model_tokenizer.load_state_dict(torch.load('quantizer_only.pt')['model'], strict=False)
        self.model_tokenizer.eval()
        self.vit = VisionTransformer().to('cuda')
    
    def forward(self, x):
        tokens , min_encoding_indices= self.model_tokenizer(x)
        x = self.vit(tokens)
        return x,(min_encoding_indices,tokens)
    
def image_preprocess(image):
    image = np.array(image) / 255.0
    image = 2.0 * image - 1.0
    image_tensor = torch.tensor(image).unsqueeze(0)
    image_tensor = torch.einsum('nhwc->nchw', image_tensor)
    image_input = image_tensor.float().to("cuda")
    del image, image_tensor
    return image_input

def image_preprocess_list(image_list):
    image = np.array(image_list) / 255.0
    image = 2.0 * image - 1.0
    image_tensor = torch.tensor(image)
    image_tensor = torch.einsum('nhwc->nchw', image_tensor)
    image_input = image_tensor.float().to("cuda")
    del image, image_tensor
    return image_input

# image_path = "testing_images/infer_fast_541.jpg"
# image = Image.open(image_path).convert('RGB')
# weights_path = "weights/model_epoch_5.pth"
# image_tensor = image_preprocess(image=image)

# # Instantiate the model
# model = NewModel()
# model.to("cuda")

# # Load the model weights
# state_dict = torch.load(weights_path)
# model.load_state_dict(state_dict)

# # Set the model to evaluation mode
# model.eval()

# # Perform inference
# with torch.no_grad():
#     output = model(image_tensor)
#     output = nn.functional.softmax(output, dim=1)
#     print(output)
#     print(f"Predicted Class : {torch.argmax(output, dim=1).item()}")

def classify(img,model):
    img = image_preprocess(img)
    output = model(img)
    print(f"Output : {output}")
    output = nn.functional.softmax(output, dim=1)
    print(f"Predicted Class : {torch.argmax(output, dim=1).item()} | {output}")

def classify_comparing_indices(img_list,model):

    img_batch = image_preprocess_list(img_list)
    output , tokens = model(img_batch)
    return tokens
    
def classify_list(img_list, model):
    
    limb_names = ["left_arm", "right_arm", "left_leg", "right_leg"]
    preprocessed_images = []
    
    img_batch = image_preprocess_list(img_list)
            
    # Create a batch from the list of preprocessed images
    # Feed the batch into the model
    output = model(img_batch)
    output = nn.functional.softmax(output, dim=1)
    present_or_absent_list = []
    
    for i, limb in enumerate(limb_names):
        if torch.argmax(output[i], dim=0).item() == 0:
            present_or_absent = "ABSENT"
            present_or_absent_list.append(present_or_absent)
        else:
            present_or_absent = "PRESENT"
            present_or_absent_list.append(present_or_absent)
        
    print(f"TRAUMA | LEFT ARM : {present_or_absent_list[0]} RIGHT ARM : {present_or_absent_list[1]} LEFT LEG : {present_or_absent_list[2]} RIGHT LEG : {present_or_absent_list[3]}")
    return present_or_absent_list
    
    
def classify_and_save(img_list,model,output_folder,frame_number):
    
    limb_names = ["left_arm", "right_arm", "left_leg", "right_leg"]
    img_batch = image_preprocess_list(img_list)
            
    # Create a batch from the list of preprocessed images
    # Feed the batch into the model
    output = model(img_batch)
    output = nn.functional.softmax(output, dim=1)
    present_or_absent_list = []
    
    for i, limb in enumerate(limb_names):
        print(limb)
        if torch.argmax(output[i], dim=0).item() == 0:
            present_or_absent = "ABSENT"
            present_or_absent_list.append(present_or_absent)
            cv2.imwrite(f"{output_folder}/ABSENT_{limb}_{frame_number}.jpg",img_list[i])
        else:
            present_or_absent = "PRESENT"
            present_or_absent_list.append(present_or_absent)
            cv2.imwrite(f"{output_folder}/PRESENT_{limb}_{frame_number}.jpg",img_list[i])
        
    print(f"TRAUMA | LEFT ARM : {present_or_absent_list[0]} RIGHT ARM : {present_or_absent_list[1]} LEFT LEG : {present_or_absent_list[2]} RIGHT LEG : {present_or_absent_list[3]}")
    return present_or_absent_list
    
    
def classify_2(img,model):
    img = image_preprocess(img)
    output, (min_encoding_indices,tokens) = model(img)
    # print(f"Tokens : {tokens}")
    # print(f"Output : {output}")
    output = nn.functional.softmax(output, dim=1)
    print(f"Predicted Class : {torch.argmax(output, dim=1).item()} | {output}")
    return tokens,min_encoding_indices
