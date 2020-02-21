import matplotlib.pyplot as plt
import os
def myfunc(filename,type):
    mypath=os.getcwd()
    mypath=mypath+filename
    with open(mypath, "r") as fp:
        data = fp.read().split("\n")
        ind_list = []
        loss_list = []
        for i in data:
            val = i.split(',')
            if val[0]=="":
                continue;
            ind_list.append(val[0])
            loss_list.append(float(val[1]))

    fig = plt.plot(loss_list)
    plt.title(type)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.show()

def main():
    myfunc("\MLP_loss.txt","MLP")
    myfunc("\CNN_loss.txt","Conv net")
if __name__=="__main__":
    main()
