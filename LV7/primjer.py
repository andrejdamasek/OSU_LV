from sklearn . cluster import KMeans
import numpy as np
# podatkovni primjeri
X = np.array ( [ [9 , 1 ] , [3 , 2 ] , [3 , 9 ] , [4 , 8 ] , [8 , 2 ] ,
[7 , 4 ] , [9 , 7 ] , [1 , 4 ] , [8 , 7 ] , [1 , 1 ] ] )
# inicijalizacija algoritma K srednjih vrijednosti
km = KMeans ( n_clusters =3 , init = 'random', n_init =5 , random_state = 0 )
# pokretanje grupiranja primjera
km . fit (X)
# dodijeljivanje grupe svakom primjeru
labels = km.predict (X)
print(labels)