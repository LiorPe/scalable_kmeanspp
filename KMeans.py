import random
import sys
import numpy as np
import time

def dist(v1, v2):
    return np.linalg.norm(v1-v2)

def is_c_closet_to_x(x, c, all_clusters):
    return all(dist(x,other_cluster)>=dist(x,c) for other_cluster in all_clusters )

class KMeans:

    def get_params(self):
        pass


    def set_params(self, n_clusters=2, seeds_algo='scalable-kmeans++', update_algo=None, max_iterations=None, t=2, l=5):
        self.n_clusters = n_clusters
        self.seeds_algo = seeds_algo
        self.update_algo = update_algo
        self.max_iterations = max_iterations
        self.t = t
        self.l = l

    def fit(self, X):
        self.init_seeds(X)
        if self.max_iterations == None:
            self.max_iterations = sys.maxsize
        itteration = 0
        cluster_updated = True
        while itteration < self.max_iterations and cluster_updated:
            centroids_to_x_dict = self.group_x_by_clusters(X)
            old_centroids = self.centroids
            self.centroids = [np.mean(np.array([X[x_idx] for x_idx in centroids_to_x_dict[c_idx]]), axis=0) for c_idx in range(len(self.centroids))]
            cluster_updated =  any(not np.array_equal(self.centroids[c_idx],old_centroids[c_idx]) for c_idx in range(len(self.centroids)))
            # print(self.centroids)
        self.centroids = [mean for mean in np.sort(np.array(self.centroids))]
        self.total_itterations = itteration
        return self.predict(X)

    def get_cluster_label_for_x(self, X, centroids_to_x_dict):
        results = []
        c_idx_to_new_cluster_map = {}
        for x_idx, x in enumerate(X):
            for c_idx, x_indices_belong_to_c in centroids_to_x_dict.items():
                if x_idx in x_indices_belong_to_c:
                    if c_idx not in c_idx_to_new_cluster_map:
                        c_idx_to_new_cluster_map[c_idx] = len(c_idx_to_new_cluster_map)+1
                    results.append(c_idx_to_new_cluster_map[c_idx])
                    break
        return results

    def group_x_by_clusters(self, X):
        centroids_to_x_dict = {c_idx: set() for c_idx in range(len(self.centroids))}
        for x_idx, x in enumerate(X):
            dist_to_all_clusters = [dist(x, c) for c in self.centroids]
            min_dist = min(dist_to_all_clusters)
            closest_centroid_idx = dist_to_all_clusters.index(min_dist)
            centroids_to_x_dict[closest_centroid_idx].add(x_idx)
        return centroids_to_x_dict

    def init_seeds(self, X):
        start = time.time()
        if self.seeds_algo == 'forgy':
            self.centroids = random.sample(X, self.n_clusters)
        if self.seeds_algo == 'random':
            self.init_seeds_random(X)
        if self.seeds_algo == 'kmeans++':
            self.init_seeds_kmeanspp(X)
        if self.seeds_algo == 'scalable-kmeans++':
            self.init_seeds_scalable_kmeanspp(X)
        self.seeds_init_duration = time.time() - start

    def init_seeds_random(self, X):
        assignments = []
        for i in range(self.n_clusters):
            assignments.append([])
        for i in range(len(X)):
            randint = random.randint(0, self.n_clusters - 1)
            assignments[randint].append(X[i])
        if any(len(assignments[n]) == 0 for n in range(len(assignments))):
            print('recursive')
            self.init_seeds(X)
        self.centroids = []
        for n in range(self.n_clusters):
            self.centroids.append(np.mean(np.array(assignments[n])))


    def init_seeds_kmeanspp(self, X, W = None):
        if W== None:
            W = [1] * len(X)
        centroids_indices = [np.random.choice(len(X), 1, p=[w / sum(W) for w in W])[0]]
        for k in range(self.n_clusters-1):
            x_dist_to_closest_centroids = [W[i]*min(dist(X[i],X[c])**2 for c in centroids_indices) for i in range(len(X))]
            x_probs = [d/sum(x_dist_to_closest_centroids) for d in x_dist_to_closest_centroids]
            try:
                centroids_indices.append(np.random.choice(len(X), 1, p=x_probs)[0])
            except:
                print(x_probs)
        self.centroids  = list([X[i] for i in centroids_indices])


    def predict(self, X):
        centroids_to_x_dict = self.group_x_by_clusters(X)
        results = self.get_cluster_label_for_x(X, centroids_to_x_dict)
        return results

    def init_seeds_scalable_kmeanspp(self, X):
        B_indices = self.overseeding_scalable_kmeanspp(X)
        B = [X[i] for i in B_indices]
        W = self.seeding_scalable_kmeanspp(B, X)
        self.init_seeds_kmeanspp(X=B,W=W)

    def overseeding_scalable_kmeanspp(self, X):
        c_indices = [random.randint(0, len(X) - 1)]
        for i in range(self.t):
            c_tag_indices = []
            x_dist_to_closest_centroids = [min(dist(X[x_idx],X[c])**2 for c in c_indices) for x_idx in range(len(X))]
            x_probs = [min (1, self.l*d/sum(x_dist_to_closest_centroids)) for d in x_dist_to_closest_centroids]
            for x_idx in range(len(X)):
                if x_probs[x_idx] >= random.uniform(0, 1):
                    c_tag_indices.append(x_idx)
            c_indices += c_tag_indices
        return c_indices

    def seeding_scalable_kmeanspp(self, B, X):
        calculated_x_indices = set()
        W = []
        for b in B:
            points_closest_to_c =[x_idx for x_idx in range(len(X)) if is_c_closet_to_x(x=X[x_idx], c=b, all_clusters=B) and x_idx not in calculated_x_indices]
            W.append(len(points_closest_to_c))
            calculated_x_indices.update(points_closest_to_c)
        return W



