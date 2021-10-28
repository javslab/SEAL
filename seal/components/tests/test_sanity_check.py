# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.ensemble import IsolationForest
# from sklearn.model_selection import ShuffleSplit

# from seal.components import SanityCheck


# def test_sanity_check():
#     X, y = make_classification(100, 20)
    
#     X = pd.DataFrame(X)
#     y = pd.Series(y)
    
#     cross_val_strategy = ShuffleSplit(4)
    
#     cross_val_folds = [(train, test) for train, test in cross_val_strategy.split(X)]
    
#     sanity_checker = SanityCheck(
#         outliers=[("TestOutlier1", IsolationForest())],
#         drifts = [("TestDrift1", IsolationForest())]
#     )
    
#     sanity_checker(X, y, cross_val_folds)
    
#     assert hasattr(sanity_checker, "result_")
    