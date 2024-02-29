from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

class OneHot():
    
    def __init__(self) -> None:
        self.onehot_encoder = OneHotEncoder()
        
    def fit_categories(self, categories) -> None:
        temp = np.reshape(categories, (-1, 1))
        self.onehot_encoder.fit(temp)
        
    def encode(self, category) -> np.ndarray:
        temp = np.reshape(category, (-1, 1))
        encoded = self.onehot_encoder.transform(temp)
        return encoded.toarray()
    
    def decode(self, encoded):
         return self.onehot_encoder.inverse_transform(encoded)
    
    
    
if __name__ == "__main__":

    
    encoder = OneHot()
    categories = ["apple", "pear", "banana", "mustard", "cow", "house", "fart", "retard", "fuck_face"]
    encoder.fit_categories(categories)
    
    print(encoder.encode(["banana", "retard"]))
    print(encoder.decode([[0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]]))

    