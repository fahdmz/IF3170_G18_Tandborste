def confusion_matrix(y_true, y_pred):
    raise NotImplementedError


def accuracy(y_true, y_pred):
    raise NotImplementedError


def precision(y_true, y_pred):
    raise NotImplementedError


def recall(y_true, y_pred):
    raise NotImplementedError


def f1_score(y_true, y_pred):
    raise NotImplementedError


def roc_auc(y_true, y_score):
    raise NotImplementedError


def classification_report(y_true, y_pred):
    raise NotImplementedError


def misclassification_cost(y_true, y_pred, fp_cost=2.0, fn_cost=1.0):
    raise NotImplementedError
