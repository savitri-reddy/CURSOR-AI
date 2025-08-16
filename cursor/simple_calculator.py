import tkinter as tk
from tkinter import ttk
import math

class SimpleCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Calculator")
        self.root.geometry("300x400")
        self.root.resizable(False, False)
        
        # Variables
        self.current_expression = ""
        self.result_var = tk.StringVar()
        self.result_var.set("0")
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Display
        display_frame = tk.Frame(self.root)
        display_frame.pack(pady=10, padx=10, fill="x")
        
        self.display = tk.Entry(display_frame, textvariable=self.result_var, 
                               font=("Arial", 20), justify="right", state="readonly")
        self.display.pack(fill="x")
        
        # Buttons frame
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Button layout
        buttons = [
            ('C', 0, 0), ('⌫', 0, 1), ('(', 0, 2), (')', 0, 3),
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('÷', 1, 3),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('×', 2, 3),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3),
            ('0', 4, 0), ('.', 4, 1), ('=', 4, 2), ('+', 4, 3),
        ]
        
        # Create buttons
        for (text, row, col) in buttons:
            button = tk.Button(buttons_frame, text=text, font=("Arial", 14),
                             width=5, height=2, command=lambda t=text: self.button_click(t))
            button.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
        
        # Configure grid weights
        for i in range(5):
            buttons_frame.grid_rowconfigure(i, weight=1)
        for i in range(4):
            buttons_frame.grid_columnconfigure(i, weight=1)
    
    def button_click(self, value):
        if value == "C":
            self.current_expression = ""
            self.result_var.set("0")
        elif value == "⌫":
            self.current_expression = self.current_expression[:-1]
            self.result_var.set(self.current_expression if self.current_expression else "0")
        elif value == "=":
            try:
                # Replace symbols with operators
                expression = self.current_expression.replace('×', '*').replace('÷', '/')
                result = eval(expression)
                self.result_var.set(str(result))
                self.current_expression = str(result)
            except:
                self.result_var.set("Error")
                self.current_expression = ""
        else:
            self.current_expression += value
            self.result_var.set(self.current_expression)

def main():
    root = tk.Tk()
    app = SimpleCalculator(root)
    root.mainloop()

if __name__ == "__main__":
    main() 