"""
ISOM5240 Storytelling Application
----------------------------------
Upload an image → generate a caption → expand into a children's story →
convert to audio → play in the browser.
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import io

# ---------- Page config ----------
st.set_page_config(page_title="Magic Story Teller", page_icon="📖")

# ---------- Model loading (cached so it loads only once) ----------
@st.cache_resource
def load_captioner():
    """Load the image-captioning pipeline from Hugging Face."""
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

@st.cache_resource
def load_story_generator():
    """Load flan-T5 with the correct text2text-generation task."""
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
    )

# ---------- Core functions ----------
def generate_caption(image: Image.Image) -> str:
    """Return a short caption describing the uploaded image."""
    captioner = load_captioner()
    result = captioner(image)
    return result[0]["generated_text"]

def generate_story(caption: str, min_words: int = 50, max_words: int = 100) -> str:
    """Expand the caption into a 50–100 word child-friendly story."""
    generator = load_story_generator()

    best_story = ""

    for attempt in range(3):
        prompt = (
            f"Write a short, cheerful story for children aged 3 to 10. "
            f"The story must be about this picture: {caption}. "
            f"Only mention friendly characters and happy events. "
            f"Do not include anything scary, violent, romantic, or adult."
        )

        output = generator(
            prompt,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
        )[0]["generated_text"]

        story = output.strip()
        story = story.replace("Story:", "").strip()

        # Deduplicate sentences
        sentences = [s.strip() for s in story.split(".") if s.strip()]
        cleaned = []
        for s in sentences:
            if s in cleaned:
                break
            cleaned.append(s)
        story = ". ".join(cleaned)
        if story and not story.endswith("."):
            story += "."

        # Trim
        words = story.split()
        if len(words) > max_words:
            story = " ".join(words[:max_words]).rsplit(".", 1)[0] + "."

        if len(story.split()) > len(best_story.split()):
            best_story = story

        if len(story.split()) >= min_words:
            return story

    return best_story
    
def text_to_speech(text: str) -> bytes:
    """Convert text to MP3 audio bytes using gTTS."""
    tts = gTTS(text=text, lang="en", slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()

# ---------- Streamlit UI ----------
def main():
    st.title("📖 Magic Story Teller")
    st.write("Upload a picture and I'll tell you a story about it!")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your picture", use_container_width=True)

        with st.spinner("Looking at your picture..."):
            caption = generate_caption(image)
        st.info(f"**What I see:** {caption}")

        with st.spinner("Writing a story..."):
            story = generate_story(caption)
        st.success("**Here is your story!**")
        st.write(story)

        with st.spinner("Recording the story..."):
            audio_bytes = text_to_speech(story)
        st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    main()
