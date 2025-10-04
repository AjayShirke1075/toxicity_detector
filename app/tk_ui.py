# app/tk_ui.py
import tkinter as tk
from tkinter import scrolledtext, messagebox
from src.predict import predict_text

def on_check():
    """
    Called when "Check Toxicity" button is pressed.
    Fetches text, predicts toxicity, and shows results.
    """
    text = txt_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Empty", "Please enter some text to check.")
        return

    try:
        res = predict_text(text)
        out_str = (
            f"Label: {res['label']}\n"
            f"Toxicity score: {res['score']:.3f}\n\n"
            f"Cleaned text:\n{res['cleaned']}"
        )
    except Exception as e:
        out_str = f"⚠️ Error predicting toxicity: {e}"

    txt_output.configure(state='normal')
    txt_output.delete("1.0", tk.END)
    txt_output.insert(tk.END, out_str)
    txt_output.configure(state='disabled')

def on_clear():
    """
    Called when "Clear" button is pressed.
    Clears both input and output text boxes.
    """
    txt_input.delete("1.0", tk.END)
    txt_output.configure(state='normal')
    txt_output.delete("1.0", tk.END)
    txt_output.configure(state='disabled')

# Main window
root = tk.Tk()
root.title("Toxicity Detector - Simple UI")
root.geometry("720x520")

# Input label and box
lbl_input = tk.Label(root, text="Enter text to analyze:")
lbl_input.pack(anchor='w', padx=8, pady=(8,0))
txt_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=12)
txt_input.pack(padx=8, pady=6)

# Buttons frame
btn_frame = tk.Frame(root)
btn_frame.pack(padx=8, pady=6, anchor='w')

btn_check = tk.Button(btn_frame, text="Check Toxicity", command=on_check)
btn_check.pack(side='left', padx=(0,6))

btn_clear = tk.Button(btn_frame, text="Clear", command=on_clear)
btn_clear.pack(side='left')

# Output label and box
lbl_output = tk.Label(root, text="Result:")
lbl_output.pack(anchor='w', padx=8, pady=(8,0))
txt_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=12, state='disabled')
txt_output.pack(padx=8, pady=6)

# Start the Tkinter main loop
root.mainloop()
