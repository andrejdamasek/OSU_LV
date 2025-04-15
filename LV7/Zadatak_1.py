import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering
from matplotlib.colors import ListedColormap


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

#generiranje podatkovnih primjera
X = generate_data(500,5)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()
#zad1----------------------------------------------------------------------------------------
#ako je flagc 1 onda imamo 3 grupe
#ako je flagc 2 onda imamo 3 grupe koje su postavljenje dijagonalno 
#ako je flagc 3 onda 4 grupe s time da su dvije rasprsene i dvije sabite
#ako je flagc 4 onda imam 2 grupe s time da su grupe kruznice sa odredenim radijusom
#ako je flagc 5 onda 2 grupe koje su oblika polu kruznice (mjeseca)

#zad2 i 3----------------------------------------------------------------------------------------
# generiranje podatkovnih primjera

km = KMeans(n_clusters=3, init="k-means++", n_init=5, random_state=0)

flag_c = 1

for i in range(5):
    X = generate_data(500, flag_c)
    km.fit(X)
    labels = km.predict(X)
    

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("podatkovni primjeri")
    plt.show()
    flag_c = flag_c + 1

#primjecujem da ako je broj K=2 da je rezultat neka linearna funkcija koja odvaja grupe,
#ako je K=3 rezultat za flagc 1, 2 i 3 je dobar, a za flagc 4 i 5 los, gdje je za grupe u krugu ili polumjesecu samo povućena neka linija koja jednoliko to dijeli
# ako je k=4 flagc 1, 2 ,4 i 5 je lose, za flagc 3 je odlicno, takoder kod primjera 4 i 5 imamo jednoliko dijeljenje grupa(360/4) pa je 60° jedna skupina a ne po radijusu od kruznice
# za k veći od 5 ne daje dobre i na prvi pogled smislene rezultate, te recimo optimalan parametar za sve primjere bi bio K=3


