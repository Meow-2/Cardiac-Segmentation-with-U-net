from utils.save_load import load_from_file
import matplotlib.pyplot as plt
path = "./result_unetpluses_o/loss_graph/"

trainloss_list = load_from_file(path + "trainloss_list.txt")
valoss_list = load_from_file(path + "valoss_list.txt")
valdice_list = load_from_file(path + "valdice_list.txt")

figure = plt.figure(figsize=(9,3))
ax = []
for i in range(3):
    ax.append(figure.add_subplot(1,3,i+1))
# ax[1].set_title('train loss')            #设置图体，plt.title
str_list = ['train loss','val loss','dice']
val_list = [trainloss_list,valoss_list,valdice_list]
for i in range(3):
    ax[i].set_xlabel('epoch')
    ax[i].set_ylabel(str_list[i])
for i in range(3):
    ax[i].plot(range(len(val_list[i])),val_list[i])
print('Best_Epoch:',valoss_list.index(min(valoss_list))+1)
print('Best_Epoch:',valdice_list.index(max(valdice_list))+1)
plt.savefig(path + 'loss.jpg')
plt.show()