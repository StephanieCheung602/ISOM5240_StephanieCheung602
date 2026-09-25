import io
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ---------- Page Config ----------
st.set_page_config(page_title="Magic Image Storyteller", page_icon="📖")

# ---------- Model Loading ----------
@st.cache_resource(show_spinner=False)
def load_captioner():
    """Load BLIP image-captioning model."""
    try:
        return pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
        )
    except Exception as e:
        st.error(f"Failed to load image captioning model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_story_generator():
    """Load TinyStories-33M for generating child-friendly narratives."""
    try:
        return pipeline(
            "text-generation",
            model="roneneldan/TinyStories-33M",
        )
    except Exception as e:
        st.error(f"Failed to load story generation model: {e}")
        return None

# ---------- Core Helper Functions ----------
def generate_caption(image: Image.Image) -> str:
    """Extract a descriptive caption from the input image."""
    captioner = load_captioner()
    if captioner is None:
        return "a peaceful day in a beautiful scene"
    try:
        result = captioner(image)
        return result[0]["generated_text"].strip()
    except Exception as e:
        st.warning(f"Could not analyze image caption: {e}")
        return "a peaceful day in a beautiful scene"

# Safety word filter
BANNED_WORDS = {
    "kill", "killed", "killing", "death", "dead", "die", "died",
    "blood", "bloody", "violence", "violent", "war", "weapon",
    "gun", "knife", "stab", "shoot", "shot",
    "romance", "romantic", "kiss", "sex", "sexy", "naked",
    "hate", "horror", "scary", "terrified", "nightmare",
    "drug", "drugs", "alcohol", "drunk", "smoke", "smoking",
}

def is_child_safe(text: str) -> bool:
    """Return True if text is safe and contains no banned terms."""
    lowered = text.lower()
    return not any(word in lowered for word in BANNED_WORDS)

def build_fallback_story(caption: str) -> str:
    """Generate a reliable 50–100 word narrative if model output is truncated."""
    return (
        f"Once upon a time, in a bright and wonderful place, there was {caption}. "
        f"Every morning, the sun shone warmly down upon this wonderful scene. "
        f"Everyone who passed by stopped to smile and admire the gentle surroundings. "
        f"It was a day filled with quiet joy, kindness, and small magical moments. "
        f"As evening approached, peaceful calm settled over everything, leaving happy memories for all."
    )

def generate_story(caption: str, min_words: int = 50, max_words: int = 100) -> str:
    """Generate a 50–100 word story based on details extracted from the image."""
    generator = load_story_generator()
    if generator is None:
        return build_fallback_story(caption)

    prompt = f"Once upon a time, there was {caption}. One sunny day, "
    best_story = ""

    for _ in range(3):
        try:
            output = generator(
                prompt,
                max_new_tokens=130,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                pad_token_id=generator.tokenizer.eos_token_id,
            )[0]["generated_text"]

            story = output.strip()

            # Format full sentences
            sentences = [s.strip() for s in story.split(".") if s.strip()]
            cleaned = []
            for s in sentences:
                if s in cleaned:
                    break
                cleaned.append(s)
            story = ". ".join(cleaned)
            if story and not story.endswith("."):
                story += "."

            # Trim to max_words constraint
            words = story.split()
            if len(words) > max_words:
                story = " ".join(words[:max_words]).rsplit(".", 1)[0] + "."

            if not is_child_safe(story):
                continue

            word_count = len(story.split())
            if word_count > len(best_story.split()):
                best_story = story

            if min_words <= word_count <= max_words:
                return story
        except Exception:
            continue

    # Fallback if generation is out of word count range
    if best_story and len(best_story.split()) >= 35:
        return best_story
    return build_fallback_story(caption)

def text_to_speech(text: str) -> bytes:
    """Convert generated story to audio format using gTTS."""
    if not text or not text.strip():
        raise ValueError("Cannot convert empty text to speech.")
    tts = gTTS(text=text, lang="en", slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()

# ---------- Main App Layout ----------
def main():
    st.title("📖 Magic Image Storyteller")
    st.write("Upload an image to extract details, generate a narrative story, and listen to the audio!")

    uploaded_file = st.file_uploader(
        "Upload an Image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Process & display input image
            image = Image.open(uploaded_file).convert("RGB")
            image.thumbnail((512, 512))
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Step 1: Caption extraction
            with st.spinner("Extracting image details..."):
                caption = generate_caption(image)
            st.info(f"**Extracted Detail:** {caption}")

            # Step 2: Story generation (50–100 words)
            with st.spinner("Generating 50–100 word story..."):
                story = generate_story(caption)

            st.success("**Generated Story:**")
            st.write(story)
            
            # Display Word Count for Verification
            word_count = len(story.split())
            st.caption(f"📏 Story Length: **{word_count} words**")

            # Step 3: Text-to-Speech Conversion
            with st.spinner("Converting story to audio..."):
                try:
                    audio_bytes = text_to_speech(story)
                    st.audio(audio_bytes, format="audio/mp3")
                except Exception as e:
                    st.warning(f"Audio conversion failed: {e}")

        except Exception as e:
            st.error(f"An error occurred while processing the app: {e}")

if __name__ == "__main__":
    main()
