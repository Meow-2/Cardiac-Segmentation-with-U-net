import numpy as np
import matplotlib.pyplot as plt

path = "./result_unet_i/Predict/"

predict_time = np.load(path+"predict_time.npy")
predict_time = predict_time.reshape(-1)
plt.figure(figsize=(12,3))
plt.bar(range(len(predict_time)), predict_time)
plt.ylim(0,30)
plt.ylabel("ms")
plt.savefig(path + 'time_bar.jpg')
plt.show()