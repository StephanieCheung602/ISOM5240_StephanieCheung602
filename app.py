"""
ISOM5240 Storytelling Application
----------------------------------
Upload an image, generate a caption, expand into a child-safe story,
convert to audio, and play it in the browser.
"""

# ---------- Imports (MUST be at the top) ----------
import io
import streamlit as st
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS

# ---------- Page config ----------
st.set_page_config(page_title="Magic Story Teller", page_icon="📖")

# ---------- Model loading ----------
@st.cache_resource(show_spinner=False)
def load_captioner():
    """Load the BLIP processor and model directly from Hugging Face."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


@st.cache_resource(show_spinner=False)
def load_story_generator():
    """Load a tiny story model trained on children's stories only."""
    return pipeline(
        "text-generation",
        model="roneneldan/TinyStories-33M",
    )


# ---------- Core functions ----------
import torch
import numpy as np

def generate_caption(image):
    """Return a short caption describing the uploaded image."""
    processor, model = load_captioner()
    
    # 1. Process the image WITHOUT tensor conversion
    inputs = processor(images=image)
    
    # 2. Manually convert the pixel values to a PyTorch tensor
    pixel_values = np.array(inputs["pixel_values"][0])
    pixel_values = torch.from_numpy(pixel_values).unsqueeze(0)
    
    # 3. Generate the caption using the manual tensor
    out = model.generate(pixel_values)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption


# Words that must never appear in a children's story.
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


def generate_story(caption: str, min_words: int = 50, max_words: int = 100) -> str:
    """
    Expand the caption into a 50-100 word child-safe story.
    Retries up to 3 times if the output is too short or fails the safety check.
    """
    generator = load_story_generator()
    best_story = ""

    for attempt in range(3):
        prompt = f"Once upon a time, there was {caption}. "

        output = generator(
            prompt,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            pad_token_id=generator.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # TinyStories echoes the prompt back, so strip it.
        story = output.replace(prompt, "").strip()

        # Deduplicate repeated sentences.
        sentences = [s.strip() for s in story.split(".") if s.strip()]
        cleaned = []
        for s in sentences:
            if s in cleaned:
                break
            cleaned.append(s)
        story = ". ".join(cleaned)
        if story and not story.endswith("."):
            story += "."

        # Trim to max_words.
        words = story.split()
        if len(words) > max_words:
            story = " ".join(words[:max_words]).rsplit(".", 1)[0] + "."

        # Safety gate.
        if not is_child_safe(story):
            continue

        # Keep the longest safe attempt as a fallback.
        if len(story.split()) > len(best_story.split()):
            best_story = story

        if len(story.split()) >= min_words:
            return story

    return best_story


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
    st.title("📖 Magic Story Teller")
    st.write("Upload a picture and I'll tell you a story about it!")

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
        st.info(f"**What I see:** {caption}")

        with st.spinner("Writing a story..."):
            story = generate_story(caption)

        if not story or not story.strip():
            st.warning("Sorry, I couldn't write a story this time. Please try again!")
            st.stop()

        st.success("**Here is your story!**")
        st.write(story)

        with st.spinner("Recording the story..."):
            audio_bytes = text_to_speech(story)
        st.audio(audio_bytes, format="audio/mp3")


if __name__ == "__main__":
    main()
