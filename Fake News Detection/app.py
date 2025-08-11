import streamlit as st
import pickle
from streamlit.components.v1 import html

# ================== Load Model & Vectorizer ================== #
# Note: Ensure these .pkl files are in the same directory as your script
# or provide the correct path. For deployment on services like Streamlit Cloud,
# these files must be included in your repository.
try:
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please make sure 'fake_news_model.pkl' and 'vectorizer.pkl' are in the correct directory.")
    st.stop()

# ================== Custom CSS: Background, macOS style, animations ================== #
st.markdown("""
<style>
    .stApp {
        background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQZI2zDPYkqjTFeDZ61xvpZDC4q7xwcLvfVOA&s');
        background-size: cover;
        background-repeat: no-repeat;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #1a1a1a;
        min-height: 100vh;
        padding: 1rem 2rem;
    }
    .title {
        text-align: center;
        font-size: 2.8em;
        font-weight: bold;
        margin-bottom: 0.3em;
        animation: fadeIn 1s ease-in-out;
        color: white;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
    }
    .subtitle {
        text-align: center;
        font-size: 1.3em;
        margin-bottom: 2em;
        animation: fadeInUp 1s ease-in-out;
        color: white;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }
    textarea {
        border-radius: 12px !important;
        border: 1px solid #d1d1d1 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 15px;
        font-size: 1em;
        transition: all 0.3s ease;
        resize: vertical;
        min-height: 180px;
        max-width: 100%;
        width: 100%;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    textarea:focus {
        border-color: #007aff !important;
        box-shadow: 0 0 8px rgba(0,122,255,0.3) !important;
        outline: none;
    }
    .stButton button {
        background: #007aff;
        color: white;
        border-radius: 12px;
        padding: 14px 28px;
        font-size: 1.15em;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
    }
    .stButton button:hover {
        background: #0051cb;
        transform: scale(1.05);
    }
    .result-box {
        max-width: 600px;
        margin: 1.5em auto 0 auto;
        padding: 24px 30px;
        border-radius: 16px;
        text-align: center;
        font-size: 1.4em;
        animation: fadeIn 0.8s ease-in-out;
        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
        user-select: none;
    }
    .fake {
        background: #ff4d4d;
        color: white;
        font-weight: bold;
        box-shadow: 0 6px 18px rgba(255,77,77,0.5);
    }
    .real {
        background: #4caf50;
        color: white;
        font-weight: bold;
        box-shadow: 0 6px 18px rgba(76,175,80,0.5);
    }
    .warning {
        background: #ffcc00;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-top: 1em;
        animation: fadeIn 0.5s ease-in-out;
        box-shadow: 0 4px 10px rgba(255,204,0,0.5);
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(25px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ================== JavaScript for Sounds ================== #
st.markdown("""
<script>
function playSound(type) {
    let audio;
    if (type === 'fake') {
        // Using a public URL for the warning sound to ensure it works everywhere.
        audio = new Audio('https://www.myinstants.com/media/sounds/error_CDOxr0V.mp3');
    } else if (type === 'real') {
        audio = new Audio('https://cdn.pixabay.com/download/audio/2021/09/16/audio_a2fea789b9.mp3?filename=success-1-6297.mp3');
    }
    if (audio) {
        audio.play().catch(e => console.error("Audio playback failed:", e));
    }
}
</script>
""", unsafe_allow_html=True)

# ================== UI TITLE ================== #
st.markdown("<div class='title'>📰 Fake News Detection App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Type or paste a news article below, then click Predict</div>", unsafe_allow_html=True)

# ================== User Input ================== #
user_input = st.text_area("News Article Text", height=200, placeholder="Paste or type the news article here...")

# ================== Prediction ================== #
if st.button("Predict"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        
        if prediction == 1:
            st.markdown("<div class='result-box fake'>🚨 This news is <b>FAKE</b></div>", unsafe_allow_html=True)
            html("<script>playSound('fake');</script>", height=0)
        else:
            st.markdown("<div class='result-box real'>✅ This news is <b>REAL</b></div>", unsafe_allow_html=True)
            html("<script>playSound('real');</script>", height=0)
    else:
        st.markdown("<div class='warning'>⚠️ Please enter some text before predicting.</div>", unsafe_allow_html=True)
