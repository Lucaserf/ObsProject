import gzip
import numpy as np

class GzipClassifier:
    def __init__(self, data):
        self.zip_data = len(gzip.compress(data))
        self.data = data

    def get_distance(self,sample):
        zip_sample = len(gzip.compress(sample))
        sample_data = " ".join([sample,self.data])
        zip_sample_data = len(gzip.compress(sample_data))

        ncd = (zip_sample_data - min(self.zip_data, zip_sample)) / max(self.zip_data, zip_sample)

        return ncd

        
    
