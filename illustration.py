from KMeans import  KMeans

km = KMeans()
km.set_params(l=2)
X = [i for i in range (10)]
km.fit(X)