from .base_model import BaseModel
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class decisionTreeRegressorModel(BaseModel):

    def build(self):
        return DecisionTreeRegressor()
    
class decisionTreeClassifierModel(BaseModel):

    def build(self):
        return DecisionTreeClassifier()