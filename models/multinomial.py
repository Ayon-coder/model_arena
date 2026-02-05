from .base_model import BaseModel
from sklearn.naive_bayes import MultinomialNB

class multinomialModel(BaseModel):

    def build(self):
        return MultinomialNB()