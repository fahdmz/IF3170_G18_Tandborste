import pandas as pd

_TARGET_KEY = "__target__"


def clean_train(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) != len(y):
        raise ValueError("X dan y harus memiliki jumlah baris yang sama")
    X = X.reset_index(drop=True)
    y = pd.Series(y.to_numpy(), name=y.name)

    keyed = X.assign(**{_TARGET_KEY: y.to_numpy()})
    keep = ~keyed.duplicated(keep="first").to_numpy()
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    adults = pd.to_numeric(X["num_adults"], errors="coerce")
    children = pd.to_numeric(X["num_children"], errors="coerce")
    no_guest = (adults.notna() & children.notna() & ((adults + children) == 0)).to_numpy()
    X = X.loc[~no_guest].reset_index(drop=True)
    y = y.loc[~no_guest].reset_index(drop=True)
    return X, y
