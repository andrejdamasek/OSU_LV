import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#zad1----------------------------------------------------------------------------------------
n_unique_colors = len(np.unique(img_array, axis=0))
print(f"Broj različitih boja u slici: {n_unique_colors}")

#zad5----------------------------------------------------------------------------------------
for i in range(1, 7):
    img = Image.imread(f"imgs\\test_{i}.jpg")

    img = img.astype(np.float64) / 255

    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))

    img_array_aprox = img_array.copy()
    #zad2----------------------------------------------------------------------------------------
    km = KMeans(n_clusters=5, init="k-means++", n_init=5, random_state=0)
    km.fit(img_array_aprox)
    labels = km.predict(img_array_aprox)
    #zad3----------------------------------------------------------------------------------------
    centroids = km.cluster_centers_
    img_array_aprox[:, 0] = centroids[labels][:, 0]
    img_array_aprox[:, 1] = centroids[labels][:, 1]
    img_array_aprox[:, 2] = centroids[labels][:, 2]
    img_array_aprox = np.reshape(img_array_aprox, (w, h, d))
    #zad4----------------------------------------------------------------------------------------
    plt.subplot(1,2,1)
    plt.plot()   
    plt.title(f"Originalna slika test_{i}.jpg")
    plt.imshow(img)    
    plt.subplot(1,2,2)    
    plt.title(f"Promijenjena slika test_{i}.jpg")
    plt.imshow(img_array_aprox)       
    plt.show()

#ako je k=8 nema maltene razlike izmedu originala i primijenjene slike
#ako je k=5 pocinjemo lagano gubiti detalje
#ako je k=3 sada puno detalja nedostaje
#ako je k=2 sada puno detalja nedostaje te smo dobili sliku sa dvije boje

# zad6----------------------------------------------------------------------------------------
inertia_values = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img_array)
    inertia_values.append(kmeans.inertia_)

plt.plot(range(2, 10), inertia_values, marker='o')
plt.xlabel('Broj klastera (K)')
plt.ylabel('Inertnost')
plt.title('Inertnost vs. Broj klastera')
plt.show()
#optimalni K je 5
#zad7----------------------------------------------------------------------------------------
K = 5
kmeans = KMeans(n_clusters=K, random_state=0)
labels = kmeans.fit_predict(img_array)

for grupa in range(K):
    mask = (labels == grupa).astype(np.uint8)
    binarna_slika = np.reshape(mask, (w, h))

    plt.figure()
    plt.title(f"Binarna slika – Grupa {grupa+1}")
    plt.imshow(binarna_slika, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
# na slici prevladavaju 2 do 3 boje , dok su ostale boje detalji, te za glavno neko prepoznavanje oblika su nam potrebne 3 boje
