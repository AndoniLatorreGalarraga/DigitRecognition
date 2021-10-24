import numpy as nm
import random
from Modules import neuralNetwork as nn
from Modules import mnist
import tkinter as tk

trainImages, trainLabels, testImages, testLabels= mnist.getData()

def main():
    def search():
        text.delete('1.0', tk.END)
        result.delete('1.0', tk.END)
        number = int(entry.get())
        text.insert('1.0', mnist.image(testImages[number]))
        array = net.compute(testImages[number])
        list = array.T.tolist()[0]
        s = ''
        for i in range(10):
            result.insert(str(float(number)), str(i) + ': ' + str(list[i]) + '\n')
        prediction = str(list.index(max(list)))
        result.insert('11.0', 'The prediction is: ' +  prediction)
    
    net = nn.NeuralNetwork([784,30,10])
    net.load()

    window = tk.Tk()
    window.columnconfigure([0, 1, 2], minsize=28)
    window.rowconfigure([0, 1, 2, 3], minsize=28)

    title = tk.Label(text='Neural Network GUI', fg='white', bg='black')
    title.grid(row=0, column=0)
    
    text = tk.Text(fg='white', bg='black', width=28, height=28)
    text.grid(row=1, column=0)

    result = tk.Text(fg='white', bg='black', width=28, height=28)
    result.grid(row=1, column=1)

    entry = tk.Entry(fg="white", bg="black", width=5)
    entry.grid(row=2, column=0)
    
    go = tk.Button(text='GO', fg='white', bg='black', command=search)
    go.grid(row=2, column=1)

    print(entry.get())

    window.mainloop()

if __name__ == '__main__':
    main()