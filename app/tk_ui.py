# app/tk_ui.py
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from src.predict import predict_text

def on_check():
    """
    Called when "Run Analysis" button is pressed.
    Fetches text, predicts toxicity, and displays results in an executive card format.
    """
    text = txt_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Required", "Please enter payload text to analyze.")
        return

    try:
        res = predict_text(text)
        score = res['score']
        is_toxic = (res['label'] == "toxic" or score >= 0.5)
        
        status_label = "🚨 FLAGGED AS TOXIC" if is_toxic else "✅ CLEARED (NON-TOXIC)"
        status_color = "#FB7185" if is_toxic else "#34D399"
        
        lbl_status.configure(text=status_label, fg=status_color)
        lbl_confidence.configure(text=f"Confidence Score: {score:.1%}")
        
        out_str = (
            f"Verdict:           {status_label}\n"
            f"Toxicity Score:    {score:.4f} ({score:.1%})\n"
            f"Policy Threshold:  0.5000\n"
            f"Preprocessed Form: {res['cleaned']}\n"
        )
    except Exception as e:
        lbl_status.configure(text="⚠️ ERROR IN INFERENCE", fg="#F87171")
        out_str = f"Inference execution error: {e}"

    txt_output.configure(state='normal')
    txt_output.delete("1.0", tk.END)
    txt_output.insert(tk.END, out_str)
    txt_output.configure(state='disabled')

def on_clear():
    """Clears both input and output text boxes."""
    txt_input.delete("1.0", tk.END)
    txt_output.configure(state='normal')
    txt_output.delete("1.0", tk.END)
    txt_output.configure(state='disabled')
    lbl_status.configure(text="AWAITING PAYLOAD", fg="#64748B")
    lbl_confidence.configure(text="Confidence Score: --")

# Main window
root = tk.Tk()
root.title("Sentinel AI • Content Moderation Console")
root.geometry("780x620")
root.configure(bg="#090D16")

# Header Frame
header_frame = tk.Frame(root, bg="#0F172A", padx=16, pady=12, relief="flat", highlightbackground="#1E293B", highlightthickness=1)
header_frame.pack(fill="x", padx=14, pady=(12, 10))

lbl_title = tk.Label(header_frame, text="🛡️ SENTINEL AI | TRUST & SAFETY", font=("Segoe UI", 13, "bold"), fg="#F8FAFC", bg="#0F172A")
lbl_title.pack(anchor="w")

lbl_subtitle = tk.Label(header_frame, text="Automated Content Moderation & Real-Time Toxicity Scoring", font=("Segoe UI", 9), fg="#94A3B8", bg="#0F172A")
lbl_subtitle.pack(anchor="w", pady=(2, 0))

# Content Body Frame
content_frame = tk.Frame(root, bg="#090D16", padx=14)
content_frame.pack(fill="both", expand=True)

# Input Section
lbl_input = tk.Label(content_frame, text="PAYLOAD TEXT INPUT:", font=("Segoe UI", 9, "bold"), fg="#94A3B8", bg="#090D16")
lbl_input.pack(anchor='w', pady=(6, 4))

txt_input = scrolledtext.ScrolledText(
    content_frame, 
    wrap=tk.WORD, 
    width=80, 
    height=7, 
    font=("Segoe UI", 10),
    bg="#111827", 
    fg="#F1F5F9", 
    insertbackground="#3B82F6",
    relief="flat",
    highlightbackground="#1E293B",
    highlightthickness=1
)
txt_input.pack(fill="x", pady=(0, 8))

# Controls
btn_frame = tk.Frame(content_frame, bg="#090D16")
btn_frame.pack(anchor='w', pady=(2, 10))

btn_check = tk.Button(
    btn_frame, 
    text="⚡ Run Moderation Audit", 
    command=on_check,
    font=("Segoe UI", 9, "bold"),
    bg="#2563EB", 
    fg="#FFFFFF",
    activebackground="#1D4ED8",
    activeforeground="#FFFFFF",
    relief="flat",
    padx=16,
    pady=6,
    cursor="hand2"
)
btn_check.pack(side='left', padx=(0, 8))

btn_clear = tk.Button(
    btn_frame, 
    text="Clear", 
    command=on_clear,
    font=("Segoe UI", 9),
    bg="#1E293B", 
    fg="#94A3B8",
    activebackground="#334155",
    activeforeground="#FFFFFF",
    relief="flat",
    padx=14,
    pady=6,
    cursor="hand2"
)
btn_clear.pack(side='left')

# Status Bar / Verdict Frame
verdict_frame = tk.Frame(content_frame, bg="#0F172A", padx=16, pady=10, highlightbackground="#1E293B", highlightthickness=1)
verdict_frame.pack(fill="x", pady=(4, 10))

lbl_status = tk.Label(verdict_frame, text="AWAITING PAYLOAD", font=("Segoe UI", 11, "bold"), fg="#64748B", bg="#0F172A")
lbl_status.pack(side="left")

lbl_confidence = tk.Label(verdict_frame, text="Confidence Score: --", font=("Segoe UI", 9), fg="#94A3B8", bg="#0F172A")
lbl_confidence.pack(side="right")

# Output Section
lbl_output = tk.Label(content_frame, text="AUDIT TELEMETRY & REPORT:", font=("Segoe UI", 9, "bold"), fg="#94A3B8", bg="#090D16")
lbl_output.pack(anchor='w', pady=(6, 4))

txt_output = scrolledtext.ScrolledText(
    content_frame, 
    wrap=tk.WORD, 
    width=80, 
    height=8, 
    font=("Consolas", 10),
    bg="#0F172A", 
    fg="#38BDF8", 
    state='disabled',
    relief="flat",
    highlightbackground="#1E293B",
    highlightthickness=1
)
txt_output.pack(fill="both", expand=True, pady=(0, 10))

if __name__ == "__main__":
    root.mainloop()
