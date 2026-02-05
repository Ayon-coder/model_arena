from .base_model import BaseModel
from sklearn.svm import SVC

class svcModel(BaseModel):

    def build(self):
        return SVC()