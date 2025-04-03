import os
os.environ["STREAMLIT_WATCH_SUPPORT"] = "false"  # Prevent PyTorch class watch error

import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline
)
from peft import PeftModel
from evaluate import load

# ======================================
# ⚙️ Streamlit page config
# ======================================
st.set_page_config(page_title="Amazon Review Assistant", layout="wide")
st.title("🤖 Amazon Review Reply Generator")

# ======================================
# 📦 Load BERT-LoRA Sentiment Classifier
# ======================================
@st.cache_resource
def load_sentiment_model():
    base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = PeftModel.from_pretrained(base_model, "app/bert_sentiment_lora")
    tokenizer = AutoTokenizer.from_pretrained("app/bert_sentiment_lora")
    model.eval()
    return tokenizer, model

cls_tokenizer, cls_model = load_sentiment_model()
label_map = {0: "negative", 1: "positive"}

@torch.no_grad()
def predict_sentiment(text):
    inputs = cls_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = cls_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs).item()
    return label_map[label], probs.numpy().tolist()[0]

# ======================================
# 🤖 Load Flan-T5-small Generator (Local)
# ======================================
@st.cache_resource
def load_flan():
    model_path = "app/flan_t5_small"  # modèle sauvegardé depuis Colab
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

flan_pipe = load_flan()

def build_prompt(user_review, sentiment):
    return (
        f"You are a helpful Amazon assistant.\n"
        f"The customer's sentiment is {sentiment.upper()}.\n"
        f"Write a short, natural and friendly reply to the following review (max 2 sentences):\n\n"
        f"{user_review}"
    )

def generate_reply(prompt):
    response = flan_pipe(prompt)[0]["generated_text"]
    return response.strip()

# ======================================
# 📊 Load BLEU / ROUGE Evaluators
# ======================================
bleu = load("bleu")
rouge = load("rouge")

# ======================================
# 🧑‍💻 Interface
# ======================================
# ======================================
# 🧑‍💻 Interface principale
# ======================================

user_review = st.text_area("✍️ Enter a customer review:", height=150)

# Initialiser les clés dans session_state si pas encore présentes
for key in ["sentiment", "probs", "reply", "submitted"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.button("Analyze & Generate Reply"):
    with st.spinner("Analyzing sentiment and generating response..."):
        sentiment, probs = predict_sentiment(user_review)
        prompt = build_prompt(user_review, sentiment)
        reply = generate_reply(prompt)

        # Sauvegarder dans le session_state
        st.session_state.sentiment = sentiment
        st.session_state.probs = probs
        st.session_state.reply = reply
        st.session_state.submitted = True

# Si on a déjà cliqué et qu'on a une réponse
if st.session_state.submitted and st.session_state.reply:
    st.markdown(f"**🧠 Detected Sentiment:** `{st.session_state.sentiment}`")

    st.markdown("---")
    st.markdown("### 💬 AI-Generated Reply")
    st.success(st.session_state.reply)

    # Feedback section
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 This reply was helpful"):
            st.toast("✅ Thanks for your feedback!", icon="👍")
    with col2:
        if st.button("👎 This reply was not helpful"):
            st.toast("❌ Sorry to hear that!", icon="👎")

    with st.expander("📊 Sentiment Probabilities"):
        st.write({label_map[i]: f"{p:.2%}" for i, p in enumerate(st.session_state.probs)})

    # Evaluation section
    st.markdown("---")
    st.markdown("### 🧑‍💬 Paste a human-written reply (for evaluation)")
    human_reply = st.text_area("Reference Reply")

    if human_reply.strip():
        bleu_score = bleu.compute(
            predictions=[st.session_state.reply],
            references=[human_reply]
        )["bleu"]

        rouge_score = rouge.compute(
            predictions=[st.session_state.reply],
            references=[human_reply]
        )["rougeL"]

        st.markdown("### 📈 Evaluation Results")
        st.write(f"**BLEU Score:** {bleu_score:.4f}")
        st.write(f"**ROUGE-L Score:** {rouge_score:.4f}")
