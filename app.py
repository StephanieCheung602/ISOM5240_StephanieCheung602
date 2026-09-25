"""
ISOM5240 Image Description Application
--------------------------------------
Upload an image → generate a description of what is seen
→ convert to audio → play it in the browser.
"""
import io
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ---------- Page config ----------
st.set_page_config(page_title="Magic Picture Describer", page_icon="🔍")

# ---------- Model loading ----------
@st.cache_resource(show_spinner=False)
def load_captioner():
    """Use the stronger BLIP-large model for better image descriptions."""
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-large",
    )

@st.cache_resource(show_spinner=False)
def load_text_expander():
    """Small model used only to expand a short caption into a longer description."""
    return pipeline(
        "text-generation",
        model="gpt2",          # more neutral than TinyStories for pure description
    )

# ---------- Core functions ----------
def generate_caption(image: Image.Image) -> str:
    """Return a caption of the uploaded image using BLIP-large."""
    captioner = load_captioner()
    result = captioner(image, max_new_tokens=50)
    return result[0]["generated_text"].strip()

# Words that must never appear in the description.
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

def generate_description(caption: str, min_words: int = 40, max_words: int = 90) -> str:
    """
    Expand the short caption into a natural description of what is visible.
    Tries to reach ~40–90 words. Falls back to the original caption if expansion fails.
    """
    expander = load_text_expander()
    best = caption

    for _ in range(2):
        prompt = (
            f"Describe this picture clearly and factually: {caption}. "
            "Only talk about what can actually be seen in the image. "
        )
        try:
            output = expander(
                prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=expander.tokenizer.eos_token_id,
            )[0]["generated_text"]
        except Exception:
            continue

        # Remove the prompt echo
        text = output.replace(prompt, "").strip()

        # Clean repeated sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        cleaned = []
        for s in sentences:
            if s in cleaned:
                break
            cleaned.append(s)
        text = ". ".join(cleaned)
        if text and not text.endswith("."):
            text += "."

        # Limit length
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]).rsplit(".", 1)[0] + "."

        if is_child_safe(text) and len(text.split()) > len(best.split()):
            best = text

        if len(best.split()) >= min_words:
            break

    return best

def text_to_speech(text: str) -> bytes:
    """Convert text to MP3 audio bytes using gTTS."""
    if not text or not text.strip():
        raise ValueError("Cannot convert empty text to speech.")
    tts = gTTS(text=text, lang="en", slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()

# ---------- Streamlit UI ----------
def main():
    st.title("🔍 Magic Picture Describer")
    st.write("Upload a picture and I'll describe what I see in it!")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Downscale to save memory
        image = Image.open(uploaded_file).convert("RGB")
        image.thumbnail((512, 512))
        st.image(image, caption="Your picture")

        with st.spinner("Looking at your picture..."):
            caption = generate_caption(image)
        st.info(f"**Short look:** {caption}")

        with st.spinner("Writing a longer description..."):
            description = generate_description(caption)

        if not description or not description.strip():
            st.warning("Sorry, I couldn't describe the picture this time. Please try again!")
            st.stop()

        st.success("**Here is what I see:**")
        st.write(description)

        with st.spinner("Recording the description..."):
            audio_bytes = text_to_speech(description)
        st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    main()
