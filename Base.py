import tkinter
import customtkinter

from tkinter import *
from tkinter import filedialog

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("840x480")

def button_function():
    print("button pressed")

def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.txt*"),
                                                       ("all files",
                                                        "*.*")))
    button.configure(text=filename);
    # the figure that will contain the plot
  
    # list of squares
    y = [i**2 for i in range(101)]
  
    # adding the subplot
    plot1 = fig.add_subplot(111)
  
    # plotting the graph
    plot1.plot(y)
    canvas.draw()
frame = customtkinter.CTkFrame(master=app)
frame.pack(pady=15,padx=60, fill="both", expand=True)
# creating the Tkinter canvas
# containing the Matplotlib figure
heading1 = customtkinter.CTkLabel(master=frame, text="Welcome to Project Ajax.", font=("Roboto",24))
heading1.pack(pady=12,padx=10)

fig = Figure(figsize = (7, 3),
             dpi = 100)
canvas = FigureCanvasTkAgg(fig,
                           master = frame)  
canvas.draw()


# placing the canvas on the Tkinter window
canvas.get_tk_widget().pack(pady=20,padx=10)

# creating the Matplotlib toolbar
toolbar = NavigationToolbar2Tk(canvas,
                               app)
toolbar.update()

# placing the toolbar on the Tkinter window
#canvas.get_tk_widget().place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

currFileButtonText = "Select Acceleration Data"
# Use CTkButton instead of tkinter Button
button = customtkinter.CTkButton(master=frame, text=currFileButtonText, command=browseFiles)
button.pack(pady=12,padx=10)

footnote = customtkinter.CTkLabel(master=app, text="Developed by Jacintha Luo, Harrison Tigert and Joshua Gonzales.", font=("Roboto",12))
footnote.pack(pady=12,padx=10)

app.mainloop()