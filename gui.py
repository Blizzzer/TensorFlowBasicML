from tkinter import Label, Button

from network import MyNetwork
from images import show_image


class MLGUI:
    def __init__(self, master):
        self.master = master
        master.title("ML handling GUI")

        self.network = MyNetwork()

        self.label = Label(master, text="This is ML handling GUI")
        self.label.pack()

        self.load_button = Button(master, text="Load Data", command=self.load)
        self.load_button.pack()

        self.learn_button = Button(master, text="Teach Network", command=self.teach)
        self.learn_button.pack()

        self.test_button = Button(master, text="Test Network", command=self.test)
        self.test_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def load(self):
        print("I'am loading data")
        self.network.load_data()
        print("Data loading finished")

    def teach(self):
        print("I'am teaching network!")
        self.network.learn()
        print("Network teaching finished")

    def test(self):
        print("I'am testing network!")
        predictions = self.network.predict()
        show_image(predictions, self.network.test_labels, self.network.test_images)
        print("Network testing finished")
