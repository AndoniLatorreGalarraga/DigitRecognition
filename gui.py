import numpy as np
import random
from Modules import neuralNetwork as nn
from Modules import mnist
import tkinter as tk



def main():             
    def search():
        result.delete('1.0', tk.END)
        number = int(entry.get())
        image = testImagesGUI[number]
        
        array = net.compute(testImages[number])
        list = array.T.tolist()[0]
        s = ''
        for i in range(10):
            result.insert(str(float(number)), str(i) + ': ' + str(list[i]) + '\n')
        prediction = str(list.index(max(list)))
        result.insert('11.0', '\nThe network prediction is: ' +  prediction)
        result.insert('13.0', '\n\nThe dataset label is: ' +  str(testLabels[number]))

        for x in range(28*8):
            for y in range(28*8):
                b = image[x//8][y//8]
                img.put('#%02x%02x%02x' % (b, b, b), (y, x))
    
        
    testImages, testLabels, testImagesGUI= mnist.getDataGUI()

    net = nn.NeuralNetwork([784,30,10])
    net.load()

    window = tk.Tk()
    window.columnconfigure([0, 1], minsize=0)
    window.rowconfigure([0, 1, 2], minsize=0)

    title = tk.Label(text='Neural Network GUI', fg='white', bg='black')
    title.grid(row=0, column=0)

    canvas = tk.Canvas(window, width=28*8, height=28*8, bg="#000000")
    canvas.grid(row=1, column=0)
    img = tk.PhotoImage(width=28*8, height=28*8)
    canvas.create_image((28*4, 28*4), image=img, state="normal")

    result = tk.Text(fg='white', bg='black', width=28, height=14)
    result.grid(row=1, column=1)

    entry = tk.Entry(fg="white", bg="black", width=5)
    entry.grid(row=2, column=0)
    
    go = tk.Button(text='GO', fg='white', bg='black', command=search)
    go.grid(row=2, column=1)

    print(entry.get())

    window.mainloop()

if __name__ == '__main__':
    main()