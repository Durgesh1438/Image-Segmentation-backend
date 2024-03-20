import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import math

ppmm = 3.16

class LineDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Multiple Line Length Measurement")

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.image = None
        self.lines = []
        self.current_line = []

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.canvas.bind("<Button-1>", self.start_line)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_line)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image()

    def display_image(self):
        height, width, _ = self.image.shape
        self.canvas.config(width=width, height=height)
        
        pil_image = Image.fromarray(self.image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def clear_canvas(self):
        self.canvas.delete("line")
        self.lines = []
        self.current_line = []

    def start_line(self, event):
        x, y = event.x, event.y
        self.current_line = [(x, y)]

    def draw_line(self, event):
        x, y = event.x, event.y
        if self.current_line:
            self.canvas.delete("line")
            self.canvas.create_line(self.current_line[0][0], self.current_line[0][1], x, y, fill="blue", width=2, tags="line")

    def end_line(self, event):
        if self.current_line:
            x, y = event.x, event.y
            self.current_line.append((x, y))
            self.lines.append(self.current_line)
            self.current_line = []

            if len(self.lines[-1]) == 2:
                x1, y1 = self.lines[-1][0]
                x2, y2 = self.lines[-1][1]
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if hasattr(self, "length_label"):
                    self.length_label.destroy()
                self.length_label = tk.Label(self.root, text=f"Length of the line: {(length/ppmm):.2f} mm")
                self.length_label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = LineDrawer(root)
    root.mainloop()
