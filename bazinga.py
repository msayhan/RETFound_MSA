from PIL import Image as PIL_Image
from torch.utils.data import Dataset

class IDRiD_ImageDataset(Dataset):
    def __init__(self, metadata, target_column='Retinopathy grade', 
                 transforms=None, target_transforms=None
                ):
        self.metadata = metadata 
        self.target_column = target_column        
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):

        filepath = self.metadata.iloc[idx]['file_path']
        with PIL_Image.open(filepath) as img:
            if len(img.size) < 3: # if single channel, convert to RGB
                img = img.convert(mode='RGB')
            
            if self.transforms:
                img = self.transforms(img)
        
        return img, int(self.metadata.iloc[idx][self.target_column])
    
    # def get_labels(self):
    #     # return as series for ImbalancedDatasetSampler to read into a Pandas dataframe
    #     return self.metadata[self.target_column]