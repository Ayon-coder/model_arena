from .base_model import BaseModel
from sklearn.linear_model import LogisticRegression

class logisticModel(BaseModel):

    def build(self):
        return LogisticRegression()