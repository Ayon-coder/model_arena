import pandas as pd
import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer

MODEL_REGISTRY = {
    "linear": {
        "task": ["regression"],
        "sparse": True,
        "text": True
    },
    "svr": {
        "task": ["regression"],
        "sparse": True,
        "text": True
    },
    "decision_tree": {
        "task": ["regression", "classification"],
        "sparse": True,
        "text": False
    },
    "logistic": {
        "task": ["classification"],
        "sparse": True,
        "text": True
    },
    "multinomialnb": {
        "task": ["classification"],
        "sparse": True,
        "text": True
    },
    "gaussiannb": {
        "task": ["classification"],
        "sparse": False,
        "text": False
    },
    "svc": {
        "task": ["classification"],
        "sparse": True,
        "text": True
    },
    "knn": {
        "task": ["classification"],
        "sparse": False,
        "text": False
    }
}

def identify_problem(y):
    le = LabelEncoder()

    if y.dtype == object:
        return "classification", le.fit_transform(y.astype(str))
    
    if np.issubdtype(y.dtype, np.number):

        if y.nunique() <= 20:
            return "classification", y.astype(int)
        else:
            return "regression", y.astype(float)
        
    raise ValueError("Unsupported target dtype")


def create_transformer(X):

    text_cols = []
    cat_cols = []
    num_cols = X.select_dtypes(include = ["float64", "int64"]).columns
    obj_cols = X.select_dtypes(include = ['object']).columns

    for col in obj_cols:

        avg_words = X[col].dropna().astype(str).apply(lambda s: len(s.split())).mean()

        if avg_words > 1:
            text_cols.append(col)
        else:
            cat_cols.append(col)

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MaxAbsScaler())
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    text_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("flatten", FunctionTransformer(lambda x: x.ravel(), validate=False)),
            ("vectorizer", CountVectorizer())
        ]
    )

    transformer = []

    if len(num_cols) > 0:
        transformer.append(("num", num_pipe, num_cols))

    if len(text_cols) > 0:
        transformer.append(("text", text_pipe, text_cols))

    if len(cat_cols) > 0:
        transformer.append(("cat", cat_pipe, cat_cols)) 

    return transformer, text_cols    



def model_sorting(X_transformed, problem_type, has_text):
    sparse = issparse(X_transformed)
    models = []

    for model, caps in MODEL_REGISTRY.items():
        if problem_type not in caps["task"]:
            continue
        if sparse and not caps["sparse"]:
            continue
        if has_text and not caps["text"]:
            continue
        models.append(model)

    if not models:
        raise ValueError("No compatible models for given data")

    return models
        
  

def process_data(df, target_col):

    X = df.drop([target_col], axis = "columns")
    y = df[target_col]
    problem_type, y = identify_problem(y)
    transformer, text_cols = create_transformer(X)
    has_text = len(text_cols) > 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    process = ColumnTransformer(transformers=transformer)
    X_train = process.fit_transform(X_train)
    X_test = process.transform(X_test)

    models_list = model_sorting(X_train, problem_type, has_text)

    return {
        "models": models_list,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "problem_type": problem_type
    }


def collect_data_info(config):
    
    dataset_path = config["dataset_path"].strip().strip('"').strip("'")
    target_col = config["target_col"]

    df = pd.read_csv(dataset_path)

    data = process_data(df, target_col)

    return data
