"""
ISOM5240 Image Description Application
--------------------------------------
Upload an image → extract details → generate a short narrative story
→ convert to audio → play it in the browser.
"""
import io
import re
import traceback
import streamlit as st
from PIL import Image

# ---------- Page config ----------
st.set_page_config(page_title="Magic Picture Storyteller", page_icon="📖")

# ---------- Globals for models ----------
captioner = None
generator = None

def load_models():
    """Load models once, with error handling."""
    global captioner, generator
    try:
        from transformers import pipeline

        with st.spinner("Loading image captioning model..."):
            captioner = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base",
            )

        with st.spinner("Loading story generation model..."):
            generator = pipeline(
                "text-generation",
                model="roneneldan/TinyStories-33M",
            )

        return True, ""
    except Exception as e:
        return False, str(e)

# Try to load models at startup
models_ok, load_error = load_models()
if not models_ok:
    st.error("Failed to load AI models. Please check your environment and dependencies.")
    st.code(f"Error details:\n{load_error}")
    st.stop()

# ---------- Core functions ----------
def generate_caption(image: Image.Image) -> str:
    """Return a short caption of the uploaded image."""
    try:
        result = captioner(image)
        if not result:
            return "a picture with some objects and colors"
        text = result[0].get("generated_text", "")
        if not text:
            return "a picture with some objects and colors"
        return text.strip()
    except Exception:
        # Fallback caption if model fails
        return "a picture with some objects and colors"

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

def clean_story_text(text: str, max_words: int = 100) -> str:
    """Basic cleaning: remove repeated sentences and trim length."""
    if not text:
        return ""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    cleaned = []
    for s in sentences:
        if s in cleaned:
            break
        cleaned.append(s)

    text = " ".join(cleaned)
    if text and not any(text.endswith(p) for p in ".!?"):
        text += "."

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
        last_dot = text.rfind(".")
        if last_dot > 20:
            text = text[: last_dot + 1]
        else:
            text = text.rsplit(" ", 1)[0] + "."

    return text

def generate_story(caption: str, min_words: int = 50, max_words: int = 100) -> str:
    """
    Generate a short narrative story inspired by the image details.
    Falls back to a template story if the model fails.
    """
    if generator is None:
        # Model didn't load properly
        return build_fallback_story(caption)

    best = ""

    try:
        for _ in range(3):
            prompt = (
                f"Image description: {caption}\n\n"
                "Task: Write a short, child-friendly story (about 60 to 90 words) that is inspired by this picture. "
                "Use the objects, people, colors, place, and actions you can infer from the description. "
                "The story should feel like a tiny tale, but it must stay related to what could actually be in the image. "
                "Keep the language simple and engaging.\n\n"
                "Story:"
            )

            output = generator(
                prompt,
                max_new_tokens=160,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                pad_token_id=generator.tokenizer.eos_token_id,
            )[0]["generated_text"]

            # Remove prompt echo if present
            if output.startswith(prompt):
                text = output[len(prompt):].strip()
            else:
                text = output.strip()

            text = clean_story_text(text,
