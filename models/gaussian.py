from .base_model import BaseModel
from sklearn.naive_bayes import GaussianNB

class gaussianModel(BaseModel):

    def build(self):
        return GaussianNB()