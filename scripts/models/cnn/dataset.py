import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CircuitDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        
        # Read CSV with custom header handling (first line space-separated, rest comma-separated)
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Split on whitespace (handles multiple spaces/tabs)
            col_names = first_line.split()
            # Read the rest of the file with pandas, treating it as comma-separated
            self.metadata = pd.read_csv(csv_path, skiprows=1, header=None, names=col_names, encoding='utf-8')
        
        # Clean column names (strip any extra whitespace)
        self.metadata.columns = self.metadata.columns.str.strip()
        
        # Ensure 'circuit_name' column exists
        if 'circuit_name' not in self.metadata.columns:
            possible_names = ['circuit name', 'CircuitName', 'circuit']
            for name in possible_names:
                if name in self.metadata.columns:
                    self.metadata.rename(columns={name: 'circuit_name'}, inplace=True)
                    break
            else:
                print("Available columns:", self.metadata.columns.tolist())
                raise KeyError("Column 'circuit_name' not found in CSV. Please check your CSV header.")
        
        self.classes = sorted(self.metadata['circuit_name'].tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.img_paths = []
        self.labels = []
        for _, row in self.metadata.iterrows():
            name = row['circuit_name']
            img_path = os.path.join(img_dir, name + '.png')
            if os.path.exists(img_path):
                self.img_paths.append(img_path)
                self.labels.append(self.class_to_idx[name])
            else:
                print(f"Warning: Image not found for {name}")
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label