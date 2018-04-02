from tkinter import *
from tkinter import filedialog
import os

import data as data_model
import model

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class Gui():
    def __init__(self, root):
        self.root = root
        self.count = 0 # count for user press the startTrain button
        self.filename = '' # currrent filename

        self.root.title('Multiple Layer Perceptron')
        # frame1 for controller panel
        self.frame1 = Frame(self.root, height=200, width=310)
        self.fileLabel = Label(self.frame1, text='FilePath: ')
        self.fileName = Label(self.frame1, text='null')
        self.lrLabel = Label(self.frame1, text='LR: ')
        self.conditionLabel = Label(self.frame1, text='Epochs: ')
        self.hiddenLabel = Label(self.frame1, text='Hidden numbers: ')
        self.layerLabel = Label(self.frame1, text='Layer numbers: ')

        self.lrEntry = Entry(self.frame1)
        self.conditionEntry = Entry(self.frame1)
        self.fileEntry = Entry(self.frame1)
        self.hiddenEntry = Entry(self.frame1)
        self.layerEntry = Entry(self.frame1)

        self.fileButton = Button(self.root, text='Open file', command=self.openFile)
        self.runButton = Button(self.root, text='Start Training', command=self.startTrain)
        self.showTrain = Button(self.root, text='Show Train', command=self.showTrain)
        self.showTest = Button(self.root, text='Show Test', command=self.showTest)

        # frame2 for visual numbers
        self.frame2 = Frame(self.root, height=220, width=310)
        self.visualLabel = []
        self.inputLabel = []
        self.var = []
        for i in range(25):
            self.visualLabel.append(Label(self.frame2, text='0'))
            self.var.append(IntVar())
            self.inputLabel.append(Checkbutton(self.frame2, variable=self.var[i]))
        self.preddata = Button(self.frame2, text='Predict Number', command=self.predNumber)

        # frame3 for message output
        self.frame3 = Frame(self.root, height=720, width=428)
        self.messageLabel = Label(self.frame3, text='Output: ')
        self.messagebox = Text(self.frame3, width=60, height=43)
        self.scrollbar = Scrollbar(self.root, orient=VERTICAL, command=self.messagebox.yview)
        self.messagebox['yscrollcommand'] = self.scrollbar.set



        self.initLayout()

    def initLayout(self):
        # frame1
        self.frame1.place(x=0, y=0)
        self.root.geometry('1600x720') # setting for my 2015mbp 13 inches
        #self.root.attributes('-fullscreen', True)
        self.fileLabel.place(x=10, y=10)
        self.fileName.place(x=200, y=10)

        self.lrLabel.place(x=10, y=40)
        self.lrEntry.place(x=120, y=40)

        self.conditionLabel.place(x=10, y=70)
        self.conditionEntry.place(x=120, y=70)

        self.hiddenLabel.place(x=10, y=100)
        self.hiddenEntry.place(x=120, y=100)

        self.layerLabel.place(x=10, y=130)
        self.layerEntry.place(x=120, y=130)

        self.fileButton.place(x=120, y=10)
        self.runButton.place(x=110, y=160)
        self.showTrain.place(x=110, y=190)
        self.showTest.place(x=110, y=220)

        # frame2
        self.frame2.place(x=0, y=390)
        for i in range(5):
            for j in range(5):
                self.visualLabel[i * 5 + j].place(x=20 + j * 28, y=10 + 30 * i)
                self.inputLabel[i * 5 + j].place(x=20 + 140 + j * 28, y=10 + 30 * i) # 140 is the gap between visualLabel and inputLabel
        self.preddata.place(x=100, y=150)

        # frame3
        self.frame3.place(x=350, y=0)
        self.messageLabel.place(x=0, y=10)
        self.messagebox.place(x=0, y=30)
        self.scrollbar.pack(side='right', fill='y')



    def openFile(self):
        path = os.getcwd()
        filename = filedialog.askopenfile(initialdir=path)
        self.filename = filename.name
        name = filename.name.split('/')[-1]
        self.fileName.config(text=name)

    def startTrain(self):
        train, label, test, test_label, max = data_model.loadDataset(self.filename)
        mlp = model.MLP(float(self.lrEntry.get()), len(train[0]), int(self.hiddenEntry.get()), int(self.layerEntry.get()), label[0].shape[1])
        for epoch in range(int(self.conditionEntry.get())):
            random_index = np.arange(len(train))
            np.random.shuffle(random_index)
            for i in range(len(train)):
                index = random_index[i]
                mlp.forward(train[index])
                mlp.backpropagate(label[index])
        tp = mlp.precision(train, label)
        rmse = mlp.rmse(train, label)
        self.train = train
        self.label = label
        self.test = test
        self.test_label = test_label

        self.messagebox.insert(END, '#')
        self.messagebox.insert(END, self.count)
        self.messagebox.insert(END, '\n')
        self.messagebox.insert(END, 'Train Precision: ')
        self.messagebox.insert(END, tp)
        self.messagebox.insert(END, '\n')

        if not test == []:
            rmse = mlp.rmse(train, label)
            p = mlp.precision(test, test_label)
            self.messagebox.insert(END, 'Test Precision: ')
            self.messagebox.insert(END, p)
            self.messagebox.insert(END, '\n')
        self.messagebox.insert(END, 'RMSE value: ')
        self.messagebox.insert(END, rmse)
        self.messagebox.insert(END, '\n')
        self.mlp = mlp
        self.count += 1

    def showTrain(self):
        plt.close()
        plt.title('Train two dimension result')
        if not int(self.hiddenEntry.get()) == 2:
            return
        data_transpose = []
        colors = []
        for i in range(self.label[0].shape[1]):
            colors.append((np.random.rand(),np.random.rand(),np.random.rand()))
        for i in range(len(self.train)):
            self.mlp.forward(self.train[i])
            transpose = self.mlp.layerOuts[-2]
            data_transpose.append(transpose)
        for i in range(len(self.train)):
            plt.plot(data_transpose[i][0][0], data_transpose[i][0][1], color=colors[self.label[i].argmax()], marker='.')
        for i in range(self.label[0].shape[1]):
            x = np.arange(0,2)
            y = -(self.mlp.WList[-1][1][i] / self.mlp.WList[-1][2][i] ) * x + self.mlp.WList[-1][0][i] / self.mlp.WList[-1][2][i]
            plt.plot(x,y,color=colors[i])

    def showTest(self):
        plt.close()
        plt.title('Test two dimension result')
        if not int(self.hiddenEntry.get()) == 2:
            return
        data_transpose = []
        colors = []
        if self.test == []:
            self.test = self.train
            self.test_label = self.label
            print("test = []")
        for i in range(self.test_label[0].shape[1]):
            colors.append((np.random.rand(),np.random.rand(),np.random.rand()))
        for i in range(len(self.test)):
            self.mlp.forward(self.test[i])
            transpose = self.mlp.layerOuts[-2]
            data_transpose.append(transpose)
        for i in range(len(self.test)):
            plt.plot(data_transpose[i][0][0], data_transpose[i][0][1], color=colors[self.test_label[i].argmax()], marker='.')
        for i in range(self.test_label[0].shape[1]):
            x = np.arange(0,2)
            y = -(self.mlp.WList[-1][1][i] / self.mlp.WList[-1][2][i] ) * x + self.mlp.WList[-1][0][i] / self.mlp.WList[-1][2][i]
            plt.plot(x,y,color=colors[i])

    def predNumber(self):
        input = []
        for i in range(25):
            input.append(self.var[i].get())
        #self.numberList.insert(END, input)
        newdata = np.array(input, dtype=np.float64)
        newdata = newdata.reshape([1, -1])
        self.mlp.forward(newdata)
        output = self.mlp.layerOuts[-1].argmax()
        for i in self.visualLabel[:]:
            i.config(text='0')
        if output == 0:
            for i in self.visualLabel[0:5]:
                i.config(text='1')
            for i in self.visualLabel[20:]:
                i.config(text='1')
            for i in self.visualLabel[::5]:
                i.config(text='1')
            for i in self.visualLabel[4::5]:
                i.config(text='1')
        elif output == 1:
            for i in self.visualLabel[2::5]:
                i.config(text='1')
        elif output == 2:
            for i in self.visualLabel[0:5]:
                i.config(text='1')
            for i in self.visualLabel[4:15:5]:
                i.config(text='1')
            for i in self.visualLabel[10:15]:
                i.config(text='1')
            for i in self.visualLabel[10:21:5]:
                i.config(text='1')
            for i in self.visualLabel[20:]:
                i.config(text='1')
        elif output == 3:
            for i in self.visualLabel[0:5]:
                i.config(text='1')
            for i in self.visualLabel[10:15]:
                i.config(text='1')
            for i in self.visualLabel[20:]:
                i.config(text='1')
            for i in self.visualLabel[4::5]:
                i.config(text='1')
if __name__ == '__main__':
    t = Tk()
    mygui = Gui(t)
    mygui.root.mainloop()



