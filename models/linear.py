from .base_model import BaseModel
from sklearn.linear_model import LinearRegression

class linearModel(BaseModel):

    def build(self):
        return LinearRegression()