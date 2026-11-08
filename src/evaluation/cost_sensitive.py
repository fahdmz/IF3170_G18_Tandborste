def tune_threshold(y_true, y_score, fp_cost=2.0, fn_cost=1.0):
    raise NotImplementedError


def bayes_optimal_threshold(fp_cost=2.0, fn_cost=1.0):
    raise NotImplementedError


def cost_sensitive_class_weight(fp_cost=2.0, fn_cost=1.0):
    raise NotImplementedError
