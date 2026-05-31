# =============================================================================
#  chatbot.py  —  FAQ Chatbot using NLTK, scikit-learn, and Streamlit
#  Author  : Your Name
#  Project : NLP FAQ Chatbot — College Internship Project
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import re
import string

# ── Third-party libraries ─────────────────────────────────────────────────────
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Download required NLTK data (runs once, cached after that) ────────────────
nltk.download("punkt",       quiet=True)
nltk.download("stopwords",   quiet=True)
nltk.download("wordnet",     quiet=True)
nltk.download("omw-1.4",     quiet=True)
nltk.download("punkt_tab",   quiet=True)


# =============================================================================
#  1. FAQ DATASET
#     Each entry is a dict with a "question" (used for matching) and
#     an "answer" (shown to the user).
# =============================================================================
FAQ_DATA = [
    {
        "question": "What is artificial intelligence?",
        "answer": (
            "Artificial Intelligence (AI) is the simulation of human intelligence "
            "by machines. It includes tasks like learning, reasoning, problem-solving, "
            "perception, and language understanding."
        ),
    },
    {
        "question": "What is machine learning?",
        "answer": (
            "Machine Learning is a subset of AI where algorithms learn patterns "
            "from data and improve their performance over time without being "
            "explicitly programmed for each task."
        ),
    },
    {
        "question": "What is deep learning?",
        "answer": (
            "Deep Learning is a subset of machine learning that uses neural networks "
            "with many layers (hence 'deep') to model complex patterns in data, "
            "especially useful for images, audio, and text."
        ),
    },
    {
        "question": "What is natural language processing?",
        "answer": (
            "Natural Language Processing (NLP) is a branch of AI that enables "
            "computers to understand, interpret, and generate human language. "
            "Examples include chatbots, translation, and sentiment analysis."
        ),
    },
    {
        "question": "What is a neural network?",
        "answer": (
            "A neural network is a computing system inspired by the human brain. "
            "It consists of layers of interconnected nodes (neurons) that process "
            "data and learn to recognise patterns."
        ),
    },
    {
        "question": "What is TF-IDF?",
        "answer": (
            "TF-IDF stands for Term Frequency–Inverse Document Frequency. "
            "It is a numerical statistic used in NLP to reflect how important "
            "a word is to a document relative to a collection of documents."
        ),
    },
    {
        "question": "What is cosine similarity?",
        "answer": (
            "Cosine similarity measures the cosine of the angle between two vectors. "
            "In NLP it is used to compare text documents: a score of 1 means identical "
            "and 0 means completely different."
        ),
    },
    {
        "question": "What is Python?",
        "answer": (
            "Python is a high-level, interpreted, general-purpose programming language "
            "known for its simple syntax and readability. It is widely used in data "
            "science, web development, and automation."
        ),
    },
    {
        "question": "What is Streamlit?",
        "answer": (
            "Streamlit is an open-source Python library that makes it easy to build "
            "and share beautiful web applications for machine learning and data science, "
            "all in pure Python with no front-end experience needed."
        ),
    },
    {
        "question": "What is scikit-learn?",
        "answer": (
            "Scikit-learn is a free Python library for machine learning. It provides "
            "simple and efficient tools for data mining, data analysis, and building "
            "ML models, built on NumPy, SciPy, and matplotlib."
        ),
    },
    {
        "question": "What is NLTK?",
        "answer": (
            "NLTK (Natural Language Toolkit) is a leading Python platform for working "
            "with human language data. It provides libraries for tokenisation, "
            "stemming, tagging, parsing, and more."
        ),
    },
    {
        "question": "What is tokenisation?",
        "answer": (
            "Tokenisation is the process of splitting text into smaller units called "
            "tokens — usually words or sentences. It is one of the first steps in "
            "any NLP pipeline."
        ),
    },
    {
        "question": "What are stopwords?",
        "answer": (
            "Stopwords are common words (like 'the', 'is', 'in') that carry little "
            "meaningful information and are usually removed during text preprocessing "
            "to improve NLP model performance."
        ),
    },
    {
        "question": "What is lemmatisation?",
        "answer": (
            "Lemmatisation reduces a word to its base or root form (called a lemma). "
            "For example, 'running' becomes 'run' and 'better' becomes 'good'. "
            "It is more accurate than stemming."
        ),
    },
    {
        "question": "What is the difference between stemming and lemmatisation?",
        "answer": (
            "Stemming chops off word endings using simple rules (fast but crude). "
            "Lemmatisation uses vocabulary and grammar rules to find the true root "
            "form (slower but more accurate)."
        ),
    },
    {
        "question": "What is a chatbot?",
        "answer": (
            "A chatbot is a software application designed to simulate conversation "
            "with humans. Chatbots can be rule-based (using patterns) or AI-based "
            "(using NLP and machine learning)."
        ),
    },
    {
        "question": "What is data preprocessing?",
        "answer": (
            "Data preprocessing is the step of cleaning and transforming raw data "
            "before feeding it to a model. In NLP this includes lowercasing, removing "
            "punctuation, stopword removal, and lemmatisation."
        ),
    },
    {
        "question": "What is overfitting in machine learning?",
        "answer": (
            "Overfitting occurs when a model learns the training data too well, "
            "including its noise, and performs poorly on new unseen data. "
            "It can be reduced using techniques like cross-validation and regularisation."
        ),
    },
    {
        "question": "What is a vector in NLP?",
        "answer": (
            "In NLP a vector is a numerical representation of text. Words or sentences "
            "are converted into arrays of numbers so that mathematical operations "
            "(like similarity comparison) can be performed on them."
        ),
    },
    {
        "question": "How does a FAQ chatbot work?",
        "answer": (
            "A FAQ chatbot works by converting both stored questions and the user's "
            "input into TF-IDF vectors, then computing cosine similarity between them. "
            "The stored question with the highest similarity score provides the answer."
        ),
    },
]


# =============================================================================
#  2. TEXT PREPROCESSING
#     Cleans and normalises text so the TF-IDF comparison is more accurate.
# =============================================================================

# Initialise the lemmatiser once (reused for every call)
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    """
    Clean and normalise a text string.

    Steps:
      1. Lowercase  — 'What' and 'what' should match
      2. Remove punctuation  — strip commas, question marks, etc.
      3. Tokenise  — split into individual words
      4. Remove stopwords  — drop 'the', 'is', 'a', etc.
      5. Lemmatise  — reduce words to their root form

    Returns a single clean string ready for TF-IDF vectorisation.
    """

    # Step 1 — Lowercase
    text = text.lower()

    # Step 2 — Remove punctuation using str.translate
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 3 — Tokenise (split into word list)
    tokens = nltk.word_tokenize(text)

    # Step 4 — Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Step 5 — Lemmatise each token
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Rejoin tokens into a single string
    return " ".join(tokens)


# =============================================================================
#  3. BUILD TF-IDF MODEL
#     Pre-process all FAQ questions and fit the TF-IDF vectoriser once
#     so it doesn't rebuild on every user message.
# =============================================================================

# Extract raw questions from the FAQ list
faq_questions = [item["question"] for item in FAQ_DATA]
faq_answers   = [item["answer"]   for item in FAQ_DATA]

# Preprocess every FAQ question
faq_questions_clean = [preprocess(q) for q in faq_questions]

# Fit TF-IDF vectoriser on the cleaned FAQ questions
#   - ngram_range=(1,2) captures single words AND two-word phrases
#   - This helps match "natural language" as a phrase, not just two separate words
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(faq_questions_clean)


# =============================================================================
#  4. MATCHING FUNCTION
#     Given a user's question, find the best matching FAQ answer.
# =============================================================================

SIMILARITY_THRESHOLD = 0.15   # Scores below this trigger the fallback response

def get_best_answer(user_input: str) -> tuple[str, float]:
    """
    Match the user's question against FAQ questions using TF-IDF + cosine similarity.

    Returns:
        answer (str)  — the best matching FAQ answer (or fallback message)
        score  (float) — the similarity score (0.0 – 1.0)
    """

    # Preprocess the user's input the same way as the FAQ questions
    user_clean = preprocess(user_input)

    # Transform the user input using the SAME fitted vectoriser
    user_vector = vectorizer.transform([user_clean])

    # Compute cosine similarity between user vector and all FAQ vectors
    similarities = cosine_similarity(user_vector, tfidf_matrix)

    # similarities is a 2-D array [[s1, s2, ... sN]]  — flatten to 1-D
    scores = similarities[0]

    # Find the index of the highest score
    best_index = scores.argmax()
    best_score = scores[best_index]

    # If the best score is below the threshold, return a fallback response
    if best_score < SIMILARITY_THRESHOLD:
        fallback = (
            "I'm sorry, I don't have an answer for that question. "
            "Please try rephrasing, or ask about AI, ML, NLP, Python, "
            "or this chatbot project."
        )
        return fallback, float(best_score)

    return faq_answers[best_index], float(best_score)


# =============================================================================
#  5. STREAMLIT USER INTERFACE
# =============================================================================

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS for a clean, professional chat look ────────────────────────────
st.markdown("""
<style>
/* Import a clean font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Page background */
.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #fafafa 100%);
}

/* Chat message bubbles */
.user-bubble {
    background: #4F46E5;
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
    line-height: 1.5;
}

.bot-bubble {
    background: white;
    color: #1f2937;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 0;
    max-width: 80%;
    border: 1px solid #e5e7eb;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.score-badge {
    font-size: 0.7rem;
    color: #9ca3af;
    margin-top: 4px;
    margin-left: 4px;
}

.chat-label-user {
    text-align: right;
    font-size: 0.72rem;
    color: #6b7280;
    margin-bottom: 2px;
    margin-right: 4px;
}

.chat-label-bot {
    font-size: 0.72rem;
    color: #6b7280;
    margin-bottom: 2px;
    margin-left: 4px;
}

/* Header banner */
.header-banner {
    background: linear-gradient(90deg, #4F46E5, #7C3AED);
    color: white;
    padding: 20px 24px;
    border-radius: 16px;
    margin-bottom: 24px;
    text-align: center;
}

.header-banner h1 { font-size: 1.6rem; margin: 0 0 4px 0; }
.header-banner p  { font-size: 0.88rem; margin: 0; opacity: 0.85; }

/* Suggestion chips */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.chip {
    background: white;
    border: 1px solid #c7d2fe;
    color: #4F46E5;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🤖 FAQ Chatbot</h1>
    <p>Ask me anything about AI, Machine Learning, NLP, or this project</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — project info ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 About This Project")
    st.info(
        "This chatbot uses **TF-IDF** vectorisation and **Cosine Similarity** "
        "to match your question with the closest FAQ entry."
    )
    st.markdown("**Tech Stack**")
    st.markdown("- 🐍 Python 3.10+")
    st.markdown("- 📝 NLTK (NLP preprocessing)")
    st.markdown("- 🔢 scikit-learn (TF-IDF + similarity)")
    st.markdown("- 🌐 Streamlit (web UI)")

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Your question is preprocessed (lowercase, remove stopwords, lemmatise)\n"
        "2. Converted to a TF-IDF vector\n"
        "3. Compared with all FAQ vectors using cosine similarity\n"
        "4. The closest match returns its answer"
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Initialise session state for chat history ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a greeting from the bot when the app first loads
    st.session_state.messages.append({
        "role": "bot",
        "content": (
            "Hello! 👋 I'm your FAQ assistant. Ask me about **AI, Machine Learning, "
            "NLP, Python, Streamlit**, or how this chatbot works!"
        ),
        "score": None,
    })

# ── Suggested questions ───────────────────────────────────────────────────────
st.markdown("**💡 Try asking:**")
suggestions = [
    "What is machine learning?",
    "How does a chatbot work?",
    "What is TF-IDF?",
    "What is lemmatisation?",
]
cols = st.columns(len(suggestions))
for col, suggestion in zip(cols, suggestions):
    if col.button(suggestion, use_container_width=True):
        # Treat a suggestion click exactly like the user typing that question
        st.session_state.pending_input = suggestion

st.markdown("---")

# ── Render all chat messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown('<div class="chat-label-user">You</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="chat-label-bot">🤖 Bot</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="bot-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("score") is not None:
            st.markdown(
                f'<div class="score-badge">Confidence: {msg["score"]:.0%}</div>',
                unsafe_allow_html=True,
            )

# ── Chat input box ────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your question here…")

# Handle suggestion-button click (stored in session state above)
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

# ── Process new input ─────────────────────────────────────────────────────────
if user_input and user_input.strip():
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip(),
        "score": None,
    })

    # Get the best answer
    answer, score = get_best_answer(user_input.strip())

    # Save bot response with the confidence score
    st.session_state.messages.append({
        "role": "bot",
        "content": answer,
        "score": score,
    })

    # Rerun so the new messages render immediately
    st.rerun()
