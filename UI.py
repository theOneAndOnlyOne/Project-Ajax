import tkinter
import customtkinter
import ajax_pipeline as ajax
import pandas as pd

from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

from urllib import request

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("1040x780")

# Create a sample DataFrame
curr_dataframe = pd.DataFrame()

def download_csv():
    global curr_dataframe
    #files = [('All Files', '*.*'), 
    #         ('Python Files', '*.py'),
    #         ('Text Document', '*.txt')]
    #file = asksaveasfile(filetypes = files, defaultextension = files)
    SAVING_PATH = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
    curr_dataframe.to_csv(SAVING_PATH)
    print('CSV file downloaded successfully!')

def browseFiles():
    global curr_dataframe
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.csv*"),
                                                       ("all files",
                                                        "*.*")))
    select_data_button.configure(text=filename);
    # the figure that will contain the plot
    new_fig, curr_dataframe = ajax.evaluate_csv(filename)
    canvas.figure = new_fig
    print(curr_dataframe)
    canvas.draw()
    #new_fig.tight_layout()

frame = customtkinter.CTkFrame(master=app)
frame.pack(pady=15,padx=60, fill="both", expand=True)
# creating the Tkinter canvas
# containing the Matplotlib figure
heading1 = customtkinter.CTkLabel(master=frame, text="Dashboard", font=("Roboto",28))
heading1.pack(pady=7,padx=20,side=TOP,anchor=NW)

heading2 = customtkinter.CTkLabel(master=frame, text="Welcome to Project Ajax! Analyze your run", font=("Roboto",14))
heading2.pack(pady=0,padx=20,anchor=NW)

fig = Figure(figsize = (8.5, 4),
             dpi = 100)
canvas = FigureCanvasTkAgg(fig,
                           master = frame)  
canvas.draw()


# placing the canvas on the Tkinter window
canvas.get_tk_widget().pack(pady=20,padx=20)

# creating the Matplotlib toolbar
toolbar = NavigationToolbar2Tk(canvas,
                               app)
toolbar.update()

# placing the toolbar on the Tkinter window
#canvas.get_tk_widget().place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

# Interactable Buttons
currFileButtonText = "Select Acceleration Data"
select_data_button = customtkinter.CTkButton(master=frame, text=currFileButtonText, command=browseFiles)

download_data_button = customtkinter.CTkButton(master=frame, text="Download CSV", command=download_csv)

download_data_button.pack(pady=12,padx=10)
select_data_button.pack(pady=12,padx=10)

footnote = customtkinter.CTkLabel(master=app, text="Developed by Jacintha Luo, Harrison Tigert and Joshua Gonzales.", font=("Roboto",12))
footnote.pack(pady=12,padx=10)

app.mainloop()