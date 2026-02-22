"""
Streamlit app for spam detection using trained Transformer model.
"""

import streamlit as st
import torch
import json

from src.transformer_model import TransformerClassifier
from src.preprocessing import TextPreprocessor

CLASS_NAMES = ['ham', 'spam']


@st.cache_resource
def load_model():
    try:
        preprocessor = TextPreprocessor.load('models/preprocessor.pkl')
        with open('models/config.json') as f:
            config = json.load(f)
        model = TransformerClassifier(**config)
        model.load_state_dict(torch.load('models/transformer_best.pth', map_location='cpu'))
        model.eval()
        metrics = None
        try:
            with open('models/metrics.json') as f:
                metrics = json.load(f)
        except FileNotFoundError:
            pass
        return model, preprocessor, metrics
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def predict(model, preprocessor, text):
    seq = torch.LongTensor([preprocessor.text_to_sequence(text)])
    with torch.no_grad():
        probs = torch.softmax(model(seq), dim=1)[0]
    idx = torch.argmax(probs).item()
    return {
        'class': CLASS_NAMES[idx],
        'confidence': probs[idx].item(),
        'probabilities': {n: probs[i].item() for i, n in enumerate(CLASS_NAMES)}
    }


st.set_page_config(page_title="MailGuard AI", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
    .result-card {border-radius: 10px; padding: 2rem; margin: 2rem 0; text-align: center;}
    .result-ham {background-color: #d1fae5; border: 3px solid #10b981;}
    .result-spam {background-color: #fecaca; border: 3px solid #ef4444;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ MailGuard AI")
st.write("AI-powered email spam detector — Transformer built from scratch")

model, preprocessor, metrics = load_model()
if model is None:
    st.error("⚠️ Model not loaded. Train the model first.")
    st.stop()

if metrics:
    st.sidebar.header("📊 Model Info")
    st.sidebar.metric("Test Accuracy", f"{metrics['test_accuracy']:.1%}")
    st.sidebar.metric("Parameters", f"{metrics['total_params']:,}")
    st.sidebar.metric("Training Time", f"{metrics['training_time_min']} min")
    if 'report' in metrics:
        st.sidebar.markdown("---")
        for cls in CLASS_NAMES:
            if cls in metrics['report']:
                r = metrics['report'][cls]
                st.sidebar.text(f"{cls}: F1={r['f1-score']:.2f} P={r['precision']:.2f} R={r['recall']:.2f}")

st.subheader("📧 Try an example")
c1, c2 = st.columns(2)
if c1.button("✅ Legitimate"):
    st.session_state.email_text = "Hi team, reminder about our meeting tomorrow at 10am. Please bring your quarterly reports. Best regards."
if c2.button("🚨 Spam"):
    st.session_state.email_text = "CONGRATULATIONS! You've won $1,000,000! Click here NOW to claim your FREE prize! Limited time offer, act fast!"

st.markdown("---")
email_text = st.text_area("Or paste your email here:", value=st.session_state.get('email_text', ''), height=200)

if st.button("🔍 Analyze Email", type="primary", use_container_width=True):
    if len(email_text.strip()) < 10:
        st.error("⚠️ Email must be at least 10 characters")
    else:
        with st.spinner("Analyzing..."):
            result = predict(model, preprocessor, email_text)

        info = {
            'ham': ('✅', 'LEGITIMATE EMAIL', 'result-ham'),
            'spam': ('🚨', 'SPAM DETECTED', 'result-spam'),
        }[result['class']]

        st.markdown(f"""
        <div class="result-card {info[2]}">
            <h1>{info[0]} {info[1]}</h1>
            <h3>Confidence: {result['confidence']:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📊 Probabilities")
        cols = st.columns(2)
        for col, name in zip(cols, CLASS_NAMES):
            p = result['probabilities'][name]
            col.metric(name.capitalize(), f"{p:.1%}")
            col.progress(p)
