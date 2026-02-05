from .base_model import BaseModel
from sklearn.svm import LinearSVR

class svrModel(BaseModel):

    def build(self):
        return LinearSVR()