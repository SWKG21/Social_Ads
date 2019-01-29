from utils import sample, sample_merge_userFeature, merge_userFeature, transform_userFeature


if __name__ == "__main__":
    sample(0.15)
    print ("Data subsampling finished")
    transform_userFeature()
    print ("User feature transformation finished")
    sample_merge_userFeature()
    print ("User feature merged with sampled data")
    merge_userFeature()
    print ("User feature combination finished")