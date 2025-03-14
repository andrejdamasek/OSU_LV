import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("road.jpg")



plt.figure()
plt.imshow(img, alpha=0.6)
plt.title("Image 60% brightnes")
plt.show()

height, width, _ = img.shape
second_quarter = img[:, width // 4 : width // 2]
plt.figure()
plt.imshow(second_quarter)
plt.title("Second quarter of a image")
plt.show()


rotated_img = np.rot90(img, k=-1)  
plt.figure()
plt.imshow(rotated_img)
plt.title("Rotated image")
plt.show()

mirrored_img = np.flipud(img)
plt.figure()
plt.imshow(mirrored_img)
plt.title("Mirrored image")
plt.show()

