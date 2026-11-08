def train_test_split(X, y, test_size=0.2, stratify=True, random_state=42):
    raise NotImplementedError


class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        raise NotImplementedError


def cross_validate(model, pipeline_factory, X, y, cv, metrics, resampler=None):
    raise NotImplementedError


def grid_search(model, param_grid, pipeline_factory, X, y, cv, scoring, resampler=None):
    raise NotImplementedError
