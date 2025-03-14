import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
num_people = data.shape[0]
print(f"Number of people: {num_people}")

height = data[:, 1]
weight  = data[:, 2]
plt.figure()
plt.scatter(height, weight , alpha=0.5)
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height and weight ratio ")
plt.show()

height_50=data[::50, 1]
weight_50=data[::50, 2]
plt.figure()
plt.scatter(height_50, weight_50, alpha=0.5, color='red')
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height and weight ratio (every 50. person)")
plt.show()

min_height = np.min(height)
max_height = np.max(height)
mean_height = np.mean(height)

print(f"Minimum height: {min_height} cm")
print(f"Maximum height: {max_height} cm")
print(f"Mean height: {mean_height:.2f} cm")


male_ind = (data[:, 0] == 1)
female_ind = (data[:, 0] == 0)
male_heights = data[male_ind, 1]
female_heights = data[female_ind, 1]

print(f"Men - Min: {np.min(male_heights)} cm, Max: {np.max(male_heights)} cm, Mean: {np.mean(male_heights):.2f} cm")
print(f"Women - Min: {np.min(female_heights)} cm, Max: {np.max(female_heights)} cm, Mean: {np.mean(female_heights):.2f} cm")
