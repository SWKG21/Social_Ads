from utils import merge_userFeature, transform_userFeature
import pandas as pd

if __name__ == "__main__":
    test = pd.read_csv("preliminary_contest_data/test_truth.csv", header=None, names=['aid', 'uid', 'label'])
    test.to_csv("data/test.csv", index=False)
    print ("test file copied")
    transform_userFeature()
    print ("User feature transformation finished")
    merge_userFeature()
    print ("User feature combination finished")