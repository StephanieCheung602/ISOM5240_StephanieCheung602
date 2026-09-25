"""
ISOM5240 Image Description Application
--------------------------------------
Upload an image → extract details → generate a short narrative story
→ convert to audio → play it in the browser.
"""
import io
import re
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ---------- Page config ----------
st.set_page_config(page_title="MagicStoryteller", page_icon="📖")

# ---------- Model loading ----------
@st.cache_resource(show_spinner=False)
def load_captioner():
    """Load BLIP via the high-level pipeline (robust on Streamlit Cloud)."""
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )

@st.cache_resource(show_spinner=False)
def load_story_generator():
    """Small model used to generate a short narrative story based on image details."""
    return pipeline(
        "text-generation",
        model="roneneldan/TinyStories-33M",
    )

# ---------- Core functions ----------
def generate_caption(image: Image.Image) -> str:
    """Return a short caption of the uploaded image."""
    captioner = load_captioner()
    result = captioner(image)
    return result[0]["generated_text"].strip()

# Words that must never appear in the story.
BANNED_WORDS = {
    "kill", "killed", "killing", "death", "dead", "die", "died",
    "blood", "bloody", "violence", "violent", "war", "weapon",
    "gun", "knife", "stab", "shoot", "shot",
    "romance", "romantic", "kiss", "kissed", "kissing",
    "sex", "sexy", "sexual", "naked", "nude",
    "hate", "hated", "horror", "scary", "terrified", "nightmare",
    "drug", "drugs", "alcohol", "drunk", "smoke", "smoking",
}

def is_child_safe(text: str) -> bool:
    """Return True if the text contains no banned words."""
    lowered = text.lower()
    return not any(word in lowered for word in BANNED_WORDS)

def clean_story_text(text: str) -> str:
    """Basic cleaning: remove repeated sentences and trim length."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    cleaned = []
    for s in sentences:
        if s in cleaned:
            break
        cleaned.append(s)

    text = " ".join(cleaned)
    # Ensure proper ending punctuation
    if text and not any(text.endswith(p) for p in ".!?"):
        text += "."

    # Trim to a reasonable length (around 60–100 words)
    words = text.split()
    if len(words) > 100:
        text = " ".join(words[:100])
        last_dot = text.rfind(".")
        if last_dot > 20:
            text = text[: last_dot + 1]
