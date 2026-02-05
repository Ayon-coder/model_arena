from .base_model import BaseModel
from sklearn.neighbors import KNeighborsClassifier

class knnModel(BaseModel):

    def build(self):
        return KNeighborsClassifier()