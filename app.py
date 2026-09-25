"""
ISOM5240 Image Description Application
--------------------------------------
Upload an image → generate a description of what is seen
→ convert to audio → play it in the browser.
"""
import io
import re
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ---------- Page config ----------
st.set_page_config(page_title="Magic Picture Describer", page_icon="🔍")

# ---------- Model loading ----------
@st.cache_resource(show_spinner=False)
def load_captioner():
    """Load BLIP via the high-level pipeline (robust on Streamlit Cloud)."""
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )

@st.cache_resource(show_spinner=False)
def load_text_expander():
    """Small model used only to expand a short caption into a longer description."""
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

# Words that often indicate storytelling / non-visual content.
STORY_INDICATORS = {
    "once upon", "there was", "there were", "he felt", "she felt", "they felt",
    "it felt", "he thought", "she thought", "they thought", "it seemed",
    "suddenly", "mysteriously", "magically", "unfortunately", "fortunately",
    "little did", "in a world", "dreamed", "wished", "hoped", "feared",
    "brave", "hero", "villain", "adventure", "journey", "quest",
    "long ago", "far away", "kingdom", "castle", "dragon", "wizard",
}

# Visual nouns/verbs that suggest the sentence is grounded in the image.
VISUAL_ANCHORS = {
    "is", "are", "has", "have", "shows", "showing", "contains", "including",
    "with", "on", "in", "at", "by", "near", "next to", "behind", "in front of",
    "wearing", "holding", "standing", "sitting", "walking", "running",
    "tree", "trees", "building", "buildings", "road", "street", "car", "cars",
    "person", "people", "man", "men", "woman", "women", "child", "children",
    "sky", "cloud", "clouds", "grass", "flower", "flowers", "water", "river",
    "sea", "lake", "mountain", "mountains", "window", "door", "table", "chair",
    "room", "house", "garden", "park", "sign", "text", "logo", "screen",
}

def is_story_like(text: str) -> bool:
    """Return True if text looks like a story rather than a description."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in STORY_INDICATORS)

def has_visual_anchor(text: str) -> bool:
    """Return True if text contains words that suggest it describes visible content."""
    lowered = text.lower()
    return any(anchor in lowered for anchor in VISUAL_ANCHORS)

def clean_story_elements(text: str) -> str:
    """Remove obvious story-like sentences or clauses."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    cleaned = []
    for s in sentences:
        if is_story_like(s):
            continue
        if not has_visual_anchor(s):
            # If sentence has no visual anchor and is long, it’s likely speculative.
            if len(s.split()) > 8:
                continue
        cleaned.append(s)
    if not cleaned:
        # If everything was removed, fall back to original but trimmed.
        return text
    return " ".join(cleaned)

def generate_description(caption: str, min_words: int = 40, max_words: int = 90) -> str:
    """
    Expand the short caption into a natural, strictly visual description.
    Avoids inventing stories, emotions, or unseen events.
    """
    expander = load_text_expander()
    best = ""

    for attempt in range(3):
        # Strongly constrained prompt to avoid storytelling.
        prompt = (
            f"Caption: {caption}\n\n"
            "Task: Write a short, factual description of what is visible in this picture. "
            "Only describe objects, people, colors, shapes, positions, and actions you can see. "
            "Do NOT tell a story, do NOT mention feelings, thoughts, time, or events outside the picture. "
            "Do NOT use phrases like 'once upon', 'there was', 'he felt', 'suddenly', or similar. "
            "Write in simple English, about 50 to 80 words.\n\n"
            "Description:"
        )

        output = expander(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            pad_token_id=expander.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # Remove prompt echo (in case model repeats it).
        if output.startswith(prompt):
            text = output[len(prompt):].strip()
        else:
            text = output.strip()

        # Clean repeated sentences.
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        cleaned = []
        for s in sentences:
            if s in cleaned:
                break
            cleaned.append(s)
        text = ". ".join(cleaned)
        if text and not text.endswith("."):
            text += "."

        # Remove obvious story-like content.
        text = clean_story_elements(text)

        # Limit length.
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words])
            # Try to cut at last full sentence.
            last_dot = text.rfind(".")
            if last_dot > 10:
                text = text[: last_dot + 1]
            else:
                text = text.rsplit(" ", 1)[0] + "."

        if not is_child_safe(text):
            continue

        # Prefer longer, visually grounded outputs.
        if len(text.split()) > len(best.split()) and has_visual_anchor(text):
            best = text

        if len(text.split()) >= min_words and has_visual_anchor(text):
            return text

    # Fallback: if model keeps drifting, build a simple structured description.
    if not best or not has_visual_anchor(best):
        # Very safe fallback: just expand the caption in a template way.
        fallback = (
            f"The picture shows: {caption}. "
            "You can see different objects, colors, and shapes in the image. "
            "The scene includes visible items and possibly people or text, depending on the photo."
        )
        words = fallback.split()
        if len(words) > max_words:
            fallback = " ".join(words[:max_words]).rsplit(".", 1)[0] + "."
        return fallback

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
        # Downscale to save memory.
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
