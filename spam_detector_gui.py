from tkinter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# 1. TRAINING IMPROVED SPAM MODEL
# -----------------------------
messages = [
    # Spam Messages
    'Win a lottery now',
    'Congratulations you won a free ticket',
    'Free entry in 2 lakhs cash prize',
    'Click here to claim your reward',
    'You have been selected as a lucky winner',
    'Claim your free vacation package now',
    'Get rich quick with this offer',
    'Exclusive deal! Claim before midnight',
    'Earn money fast with this link',
    'You have won $1000 cash prize',

    # Not Spam (Normal) Messages
    'Hello friend, how are you?',
    'Let’s meet tomorrow for lunch',
    'Are you coming to college today?',
    'Project submission is tomorrow',
    'I will call you later in the evening',
    'Please find the attached report',
    'Happy birthday! Have a great day',
    'Can we meet at 5 pm?',
    'I am on my way to the office',
    'Your marksheet is available online'
]

# Labels: 1 = Spam, 0 = Not Spam
labels = [1]*10 + [0]*10

# -----------------------------
# 2. TRAINING PROCESS
# -----------------------------
cv = CountVectorizer()
x = cv.fit_transform([msg.lower() for msg in messages])  # Convert to lowercase
model = MultinomialNB()
model.fit(x, labels)

# -----------------------------
# 3. GUI DESIGN (Tkinter)
# -----------------------------
def check_spam():
    msg = entry.get().strip().lower()
    if not msg:
        output_label.config(text="⚠️ Please enter a message!", fg="orange")
        return
    data = cv.transform([msg]).toarray()
    result = model.predict(data)[0]
    if result == 1:
        output_label.config(text="⚠️ Spam Message Detected", fg="red")
    else:
        output_label.config(text="✅ Not Spam", fg="green")

# GUI Window
root = Tk()
root.title("AI Spam Message Detector - Comprehensive Version")
root.geometry("550x400")
root.config(bg="#f5f5f5")

# Heading
Label(root, text="AI Spam Message Detector", font=("Helvetica", 18, "bold"), bg="#f5f5f5", fg="#333").pack(pady=20)

# Entry Label
Label(root, text="Enter a message to analyze:", font=("Helvetica", 12), bg="#f5f5f5").pack(pady=5)

# Text Entry
entry = Entry(root, width=60, font=("Helvetica", 12))
entry.pack(pady=5)

# Check Button
Button(root, text="Check Message", command=check_spam, font=("Helvetica", 12, "bold"),
       bg="#4CAF50", fg="white", padx=15, pady=8, relief=RAISED, borderwidth=3).pack(pady=20)

# Output Label
output_label = Label(root, text="", font=("Helvetica", 14, "bold"), bg="#f5f5f5")
output_label.pack(pady=10)

# Footer
Label(root, text="Developed by Anil Kumar | MCA (AI/ML)", font=("Helvetica", 9), bg="#f5f5f5", fg="#666").pack(side=BOTTOM, pady=5)

root.mainloop()
