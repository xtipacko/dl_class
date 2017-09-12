import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset_rnd(X,Y, rows, columns, class_num=-1):
    plt.rcParams['figure.figsize'] = (columns*1.5,rows*1.7)
    plt.rcParams['toolbar'] = 'None'
    f, ax = plt.subplots(rows, columns)

    m = X.shape[1]
    image_indices = np.random.randint(0, m, size=(rows, columns))
    for i in range(rows):
        for j in range(columns):
            img_index = image_indices[i,j]
            img = np.copy(X[:,img_index])
            img = img.reshape(3,32,32)
            img = np.swapaxes(img,0,2)
            img = np.swapaxes(img,0,1)
            axN = ax[i][j]
            axN.set_title(f'{Y[0,img_index]!s} [{img_index}]')
            axN.get_xaxis().set_visible(False)
            axN.get_yaxis().set_visible(False)
            imgplot = axN.imshow(img)

    plt.show()

def visualize_formated_img(*args):
    max_col = 15
    m = len(args)
    rows = m // max_col + 1
    if rows == 1:
        columns = m % max_col 
    else:
        columns = max_col       
    
    plt.rcParams['figure.figsize'] = (columns*1.5,rows*1.7)
    plt.rcParams['toolbar'] = 'None'
    # print(rows, columns)
    f, ax = plt.subplots(rows, columns)
    for i, img in enumerate(args):
        if columns == 1 and rows == 1:
            axN = ax
        elif rows <= 1:
            axN = ax[i]
        else:
            row = i // max_col + 1
            col = i % max_col
            axN = ax[row][col]
        axN.get_xaxis().set_visible(False)
        axN.get_yaxis().set_visible(False)
        imgplot = axN.imshow(img) #, interpolation="bicubic"


    plt.show()
