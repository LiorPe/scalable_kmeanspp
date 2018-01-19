from evaluation import evaluate
import random

# X needs to be list of arrays
X = [[random.randint(0,1000),random.randint(0,1000),random.randint(0,1000)] for i in range(1000)]

# if you don`t have labels -
labels = [random.randint(100,103) for i in range(1000)]

# determine on what setting the evaluation will be mades
settings = []
settings.append({'seeds_algo': 'forgy', 'n_clusters': 10, 'max_iterations': None, 't': 5, 'l': 10})
settings.append({'seeds_algo': 'random', 'n_clusters': 10, 'max_iterations': None, 't': 5, 'l': 10})
settings.append({'seeds_algo': 'kmeans++', 'n_clusters': 10, 'max_iterations': None, 't': 5, 'l': 10})
settings.append({'seeds_algo': 'scalable-kmeans++', 'n_clusters': 10, 'max_iterations': 100, 't': 5, 'l': 10})


evaluate(X=X,labels=labels,settings=settings)
