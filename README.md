# 🤖 FAQ Chatbot — NLP Project
### College Internship Project Documentation

---

## 📁 Folder Structure

```
faq-chatbot/
├── chatbot.py        ← Complete application (NLP logic + Streamlit UI)
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

That's it — just **one Python file** runs the entire project.

---

## ⚙️ Setup Instructions (Step by Step)

### Step 1 — Make sure Python is installed
```bash
python --version    # should print Python 3.10 or higher
```
Download Python from https://www.python.org if needed.

---

### Step 2 — Create a virtual environment (recommended)
```bash
# Navigate to the project folder
cd faq-chatbot

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

---

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

### Step 4 — Run the chatbot
```bash
streamlit run chatbot.py
```

The app will open automatically at **http://localhost:8501**

---

## ▶️ Run Command (summary)
```bash
streamlit run chatbot.py
```

---

## 📦 Dependencies Explained

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.35.0 | Builds the web chat interface |
| nltk | 3.8.1 | Tokenisation, stopword removal, lemmatisation |
| scikit-learn | 1.4.2 | TF-IDF vectorisation + cosine similarity |

No database, no API keys, no frontend code needed.

---

## 🧠 Code Explanation (Section by Section)

### Section 1 — FAQ Dataset
```python
FAQ_DATA = [
    { "question": "What is machine learning?",
      "answer": "Machine Learning is..." },
    ...
]
```
A Python list of 20 dictionaries. Each has a `question` (used for matching)
and an `answer` (shown to the user). Easy to extend — just add more entries.

---

### Section 2 — Text Preprocessing
```python
def preprocess(text):
    text = text.lower()                          # Step 1: lowercase
    text = text.translate(...punctuation...)     # Step 2: remove punctuation
    tokens = nltk.word_tokenize(text)            # Step 3: tokenise
    tokens = [w for w if w not in stopwords]     # Step 4: remove stopwords
    tokens = [lemmatizer.lemmatize(w) for w...]  # Step 5: lemmatise
    return " ".join(tokens)
```
Both FAQ questions and user input are passed through this function before
comparison, so "What IS Machine Learning??" and "machine learning" both
become "machine learn" — making the match more reliable.

---

### Section 3 — TF-IDF Model
```python
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(faq_questions_clean)
```
- TF-IDF converts each question into a numerical vector.
- `ngram_range=(1,2)` means both single words AND two-word phrases are captured.
- The vectoriser is **fitted once** at startup — not on every user message.

---

### Section 4 — Matching Function
```python
def get_best_answer(user_input):
    user_vector  = vectorizer.transform([preprocess(user_input)])
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    best_index   = similarities[0].argmax()
    best_score   = similarities[0][best_index]

    if best_score < SIMILARITY_THRESHOLD:
        return fallback_message, best_score

    return faq_answers[best_index], best_score
```
1. The user's input is transformed using the **same fitted vectoriser**.
2. Cosine similarity is computed between the user vector and all FAQ vectors.
3. The highest-scoring FAQ answer is returned.
4. If no score exceeds the threshold (0.15), a fallback message is shown.

---

### Section 5 — Streamlit UI
```python
st.set_page_config(...)          # browser tab title and icon
st.session_state.messages        # stores full chat history
st.chat_input(...)               # text box at the bottom
st.markdown(...)                 # renders HTML/CSS message bubbles
st.rerun()                       # refreshes the page after each message
```
Streamlit manages the page — no HTML, no JavaScript, no server setup.
`st.session_state` persists the conversation across reruns within the same session.

---

## 📝 Internship Report Explanation

**Project Title:** FAQ Chatbot using NLP and Machine Learning

**Objective:**
To develop an intelligent FAQ chatbot that can understand natural language
questions from users and return the most relevant pre-defined answer using
Natural Language Processing (NLP) and Machine Learning techniques.

**Technologies Used:**
- Python 3.10
- NLTK — for text preprocessing (tokenisation, stopword removal, lemmatisation)
- scikit-learn — for TF-IDF vectorisation and cosine similarity computation
- Streamlit — for building and deploying the web-based chat interface

**Methodology:**
The system follows a retrieval-based approach:
1. A dataset of 20 FAQ question-answer pairs is defined in the code.
2. All FAQ questions are preprocessed using NLTK (lowercased, punctuation removed,
   stopwords filtered out, words lemmatised to their root form).
3. A TF-IDF vectoriser (scikit-learn) is fitted on the cleaned FAQ questions,
   converting them into numerical vectors.
4. When a user submits a question, the same preprocessing pipeline is applied
   to their input, and the result is transformed into a TF-IDF vector.
5. Cosine similarity is computed between the user's vector and all FAQ vectors.
6. The FAQ with the highest similarity score returns its answer to the user.
7. If the best score falls below a defined threshold (0.15), a fallback message
   is returned to inform the user that no good match was found.
8. The entire application is served as an interactive web app using Streamlit,
   featuring a conversational chat interface with message history, suggested
   questions, and a confidence score display.

**Results:**
The chatbot successfully answers questions related to AI, ML, NLP, Python,
and project-specific topics. It handles slight variations in phrasing due
to text normalisation and returns a confidence percentage with each answer.

**Learning Outcomes:**
- Understood the NLP preprocessing pipeline end-to-end
- Learned how TF-IDF represents text as numerical vectors
- Applied cosine similarity for semantic text comparison
- Built and deployed an interactive ML web application with Streamlit
- Gained experience with session state management and UI design in Python

---

## 🎓 Viva Questions & Answers

### Q1. What is TF-IDF and why did you use it?
**Answer:** TF-IDF stands for Term Frequency–Inverse Document Frequency.
TF measures how often a word appears in a document; IDF penalises words that
appear in many documents (making common words less important). Together they
give each word a weight that reflects its importance to a specific document.
I used it because it is simple, effective, and requires no training data —
perfect for a FAQ matching system where the vocabulary is known in advance.

---

### Q2. What is cosine similarity?
**Answer:** Cosine similarity measures the cosine of the angle between two
vectors. If two vectors point in the same direction, the cosine is 1 (identical);
if they are perpendicular, it is 0 (no similarity). In text, vectors represent
word frequencies/weights. I use it to compare the user's question vector with
all FAQ vectors and find the closest match.

---

### Q3. What is the preprocessing pipeline in your project?
**Answer:** Five steps:
1. **Lowercasing** — so 'What' and 'what' are treated the same
2. **Punctuation removal** — strips question marks, commas, etc.
3. **Tokenisation** — splits the sentence into individual words
4. **Stopword removal** — removes common words like 'the', 'is', 'a'
5. **Lemmatisation** — reduces words to their root form (e.g. 'running' → 'run')
Both the FAQ questions and user input go through the same pipeline so they
are comparable.

---

### Q4. What is the difference between stemming and lemmatisation?
**Answer:** Stemming chops word endings using simple fixed rules — fast but
can produce non-words (e.g. 'studies' → 'studi'). Lemmatisation uses a
dictionary and grammar rules to find the actual root word (e.g. 'studies' →
'study'). I chose lemmatisation because it produces real words and gives
more accurate results, which improves matching quality.

---

### Q5. What is `session_state` in Streamlit?
**Answer:** By default, Streamlit reruns the entire script from top to bottom
on every user interaction. `st.session_state` is a dictionary that persists
values across these reruns within the same browser session. I use it to store
the chat message history so previous messages are not lost when the user sends
a new message.

---

### Q6. What happens when no good match is found?
**Answer:** If the highest cosine similarity score across all FAQ questions
falls below the threshold of 0.15, the `get_best_answer` function returns a
fallback message instead of a potentially wrong answer. The threshold was
chosen empirically — low enough to match reasonable paraphrases but high
enough to reject completely unrelated questions.

---

### Q7. What is `ngram_range=(1, 2)` in the TF-IDF vectoriser?
**Answer:** By default TF-IDF only looks at single words (unigrams). Setting
`ngram_range=(1, 2)` also includes two-word phrases (bigrams). This helps
the model recognise important phrases like "machine learning", "natural language",
or "cosine similarity" as single units rather than two unrelated words, which
improves matching accuracy.

---

### Q8. What type of chatbot is this — generative or retrieval-based?
**Answer:** This is a **retrieval-based** chatbot. It does not generate new
text; it selects the best pre-written answer from a fixed FAQ dataset. Generative
chatbots (like GPT) create new responses word by word using language models.
Retrieval-based chatbots are more predictable and suitable for FAQ systems
where accurate, controlled answers are required.

---

### Q9. How would you improve this chatbot for production?
**Answer:** Several improvements:
1. Replace TF-IDF with **sentence embeddings** (e.g. sentence-transformers)
   for better semantic understanding
2. Add a **database** (SQLite or Firebase) to store and manage FAQs dynamically
3. Implement **conversation context** — remember previous turns
4. Add **feedback buttons** (👍 👎) to collect data for improvement
5. Deploy on **Streamlit Cloud** or **Heroku** for public access
6. Add **admin panel** to add/edit FAQs without touching code

---

### Q10. What is the role of the `fit_transform` and `transform` methods?
**Answer:** `fit_transform` does two things at once: it **fits** the vectoriser
(learns the vocabulary and IDF weights from the FAQ questions) and then
**transforms** them into TF-IDF vectors. `transform` alone applies the
already-learned vocabulary to new text (the user's input) without re-learning.
It is important that the user's input uses `transform` (not `fit_transform`)
so it is compared in the same vector space as the FAQ questions.

---

*FAQ Chatbot • Built with Python, NLTK, scikit-learn, Streamlit • College Internship Project*
