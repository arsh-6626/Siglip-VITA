import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VisionFineTuneDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        
        self.label_mapping = {
            'Normal': '0',
            'Wound': '1', 
            'Amputation': '2',
            'Not Testable': '3',

        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_name']
        
        if image_name.startswith('/images/'):
            image_path = os.path.join(self.image_dir, image_name.replace('/images/', ''))
        elif image_name.endswith('.json'):
            base_name = image_name.rsplit('.', 1)[0]
            image_path = os.path.join(self.image_dir, base_name + '.jpeg')
            if not os.path.exists(image_path):
                image_path = os.path.join(self.image_dir, base_name + '.png')
        else:
            image_path = os.path.join(self.image_dir, image_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            print(f"Warning: Image not found: {image_path}")
        
        image = self.transform(image)
        
        labels = {
            "trauma_head": self.label_mapping.get(row['Head'], '0'),
            "trauma_torso": self.label_mapping.get(row['Torso'], '0'),
            "trauma_upper_ext": self.label_mapping.get(row['Upper Extremities'], '0'),
            "trauma_lower_ext": self.label_mapping.get(row['Lower Extremities'], '0'),
        }
        
        return image, labels
