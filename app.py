import io
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS  # ✅ All lowercase 'gtts'

# ---------- Page config ----------
st.set_page_config(page_title="Magic Picture Describer", page_icon="🔍")

# ---------- Model loading with error handling ----------
@st.cache_resource(show_spinner=False)
def load_captioner():
    """Load BLIP via the high-level pipeline."""
    try:
        return pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
        )
    except Exception as e:
        st.error(f"Failed to load image captioning model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_text_expander():
    """Small model used strictly to expand caption into visual details (no stories)."""
    try:
        return pipeline(
            "text-generation",
            model="roneneldan/TinyStories-33M",
        )
    except Exception as e:
        st.error(f"Failed to load text expansion model: {e}")
        return None

# ---------- Core functions ----------
def generate_caption(image: Image.Image) -> str:
    """Return a short factual caption of the uploaded image."""
    captioner = load_captioner()
    if captioner is None:
        return "An uploaded image."
    try:
        result = captioner(image)
        return result[0]["generated_text"].strip()
    except Exception as e:
        st.warning(f"Could not generate caption: {e}")
        return "An uploaded image."

# Words that must never appear in the description
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

def generate_description(caption: str, min_words: int = 30, max_words: int = 80) -> str:
    """
    Expand the short caption into a clear, factual visual description.
    Strictly forbids story generation or imagined scenarios.
    """
    expander = load_text_expander()
    if expander is None:
        return f"This image shows {caption}."

    best = ""

    # Strict prompt instructing the model to describe only visible facts
    prompt = (
        f"Based on the caption '{caption}', describe only the visible elements in the image. "
        "Do not tell a story or invent imagined events. Focus on colors, objects, and setting: "
    )

    for _ in range(3):
        try:
            output = expander(
                prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.3,  # Lower temperature reduces creative storytelling
                top_p=0.85,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                pad_token_id=expander.tokenizer.eos_token_id,
            )[0]["generated_text"]

            # Remove prompt echo
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

            # Truncate over max_words
            words = text.split()
            if len(words) > max_words:
                text = " ".join(words[:max_words]).rsplit(".", 1)[0] + "."

            if not is_child_safe(text):
                continue

            if len(text.split()) > len(best.split()):
                best = text

            if len(text.split()) >= min_words:
                return text
        except Exception:
            pass

    if best.strip():
        return best

    # Factual fallback if model generation fails
    return f"This image displays {caption} in clear detail."

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
    st.write("Upload a picture and I'll describe what is visible in it!")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Downscale to save memory
            image = Image.open(uploaded_file).convert("RGB")
            image.thumbnail((512, 512))
            st.image(image, caption="Your picture", use_container_width=True)

            with st.spinner("Analyzing your picture..."):
                caption = generate_caption(image)
            st.info(f"**Quick Overview:** {caption}")

            with st.spinner("Generating visual description..."):
                description = generate_description(caption)

            st.success("**Visual Description:**")
            st.write(description)

            with st.spinner("Generating audio..."):
                try:
                    audio_bytes = text_to_speech(description)
                    st.audio(audio_bytes, format="audio/mp3")
                except Exception as e:
                    st.warning(f"Could not generate audio: {e}")

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

if __name__ == "__main__":
    main()
