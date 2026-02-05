from models import *

def winner(model_acc):
    best_model = None
    min_gap = float("inf")
    verdict = {}

    for model_name, scores in model_acc.items():
        train = scores["train"]
        test = scores["test"]
        gap = abs(train - test)

        # store gap
        scores["gap"] = gap

        # pick winner
        if gap < min_gap:
            min_gap = gap
            best_model = model_name

        # diagnose model
        if train > test and gap > 0.1:
            verdict[model_name] = "overfitting"
        elif train < 0.6 and test < 0.6:
            verdict[model_name] = "underfitting"
        else:
            verdict[model_name] = "good_fit"

    return {
        "winner": best_model,
        "details": model_acc,
        "verdict": verdict
    }
 


def resolve_model(model_name, problem_type):
    if model_name == "decision_tree":
        if problem_type == "classification":
            return decisionTreeClassifierModel()
        else:
            return decisionTreeRegressorModel()

    elif model_name == "linear":
        return linearModel()

    elif model_name == "logistic":
        return logisticModel()

    elif model_name == "svr":
        return svrModel()

    elif model_name == "svc":
        return svcModel()

    elif model_name == "knn":
        return knnModel()

    elif model_name == "gaussiannb":
        return gaussianModel()

    elif model_name == "multinomialnb":
        return multinomialModel()

    else:
        raise ValueError(f"Unknown model: {model_name}")


def train(data_bundle):

    model_acc = {}

    X_train = data_bundle["X_train"]
    X_test = data_bundle["X_test"]
    y_train = data_bundle["y_train"]
    y_test = data_bundle["y_test"]

    problem_type = data_bundle["problem_type"]
    models = data_bundle["models"]

    for model_name in models:

        model_obj = resolve_model(model_name, problem_type)
        model = model_obj.build()
        model.fit(X_train, y_train)

        model_acc[model_name] = {
            "train": model.score(X_train, y_train),
            "test": model.score(X_test, y_test)
        }
    
    verdict = winner(model_acc)
    return verdict
