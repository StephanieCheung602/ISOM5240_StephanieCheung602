"""
ISOM5240 Storytelling Application
----------------------------------
Upload an image, generate a caption, expand it into a child-safe story,
convert the story to audio, and play it in the browser.
"""

import io
import re

import streamlit as st
import torch
from gtts import gTTS
from PIL import Image, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor, pipeline


# ---------- Page configuration ----------
st.set_page_config(
    page_title="Magic Story Teller",
    page_icon="📖",
    layout="centered",
)


# ---------- Model loading ----------
@st.cache_resource(show_spinner=False)
def load_captioner():
    """
    Load the BLIP image-captioning model once.
    Streamlit keeps this model in cache between reruns.
    """
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model.eval()

    return processor, model


@st.cache_resource(show_spinner=False)
def load_story_generator():
    """
    Load a small model trained on children's stories.
    """
    return pipeline(
        task="text-generation",
        model="roneneldan/TinyStories-33M",
        device=-1,  # Use CPU, which is appropriate for Streamlit Cloud.
    )


# ---------- Child-safety checks ----------
BANNED_WORDS = {
    "kill", "killed", "killing",
    "death", "dead", "die", "died",
    "blood", "bloody",
    "violence", "violent",
    "war", "weapon",
    "gun", "knife", "stab",
    "shoot", "shot",
    "romance", "romantic",
    "kiss", "kissed", "kissing",
    "sex", "sexy", "sexual",
    "naked", "nude",
    "hate", "hated",
    "horror", "scary", "terrified", "nightmare",
    "drug", "drugs",
    "alcohol", "drunk",
    "smoke", "smoking",
}


def is_child_safe(text: str) -> bool:
    """
    Return True only if text contains none of the banned terms.

    Word boundaries avoid accidental substring matches.
    For example, the check distinguishes a complete banned word
    from it merely occurring within another word.
    """
    pattern = r"\b(" + "|".join(
        re.escape(word) for word in BANNED_WORDS
    ) + r")\b"

    return re.search(pattern, text.lower()) is None


# ---------- Core application functions ----------
def generate_caption(image: Image.Image) -> str:
    """
    Generate a short caption for an uploaded image using BLIP.

    This deliberately avoids processor(..., return_tensors="pt"),
    which can trigger BatchFeature conversion errors when installed
    package versions are incompatible on Streamlit Cloud.
    """
    processor, model = load_captioner()

    image = image.convert("RGB")

    processed = processor(
        images=image,
        return_tensors=None,
    )

    pixel_values = processed["pixel_values"]

    if not isinstance(pixel_values, torch.Tensor):
        pixel_values = torch.tensor(pixel_values, dtype=torch.float32)

    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)

    pixel_values = pixel_values.to(dtype=torch.float32)

    with torch.inference_mode():
        output_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=40,
            num_beams=3,
        )

    caption = processor.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    return caption or "a happy and colourful picture"


def clean_story(text: str, prompt: str, max_words: int) -> str:
    """
    Remove the echoed prompt, remove duplicate sentences,
    and limit the story to the requested maximum word count.
    """
    story = text.replace(prompt, "", 1).strip()

    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", story)
        if sentence.strip()
    ]

    cleaned_sentences = []

    for sentence in sentences:
        if sentence not in cleaned_sentences:
            cleaned_sentences.append(sentence)

    story = ". ".join(cleaned_sentences).strip()

    if story and not story.endswith("."):
        story += "."

    words = story.split()

    if len(words) > max_words:
        story = " ".join(words[:max_words]).strip()

        if "." in story:
            story = story.rsplit(".", 1)[0].strip() + "."

        elif story:
            story += "."

    return story


def fallback_story(caption: str) -> str:
    """
    Provide a safe fallback in case the text-generation model produces
    an empty or unsuitable output.
    """
    return (
        f"One sunny day, a little child found {caption}. "
        "It looked bright and interesting. The child smiled, looked closely, "
        "and imagined a wonderful adventure. Soon, friendly helpers joined in, "
        "and everyone shared kind ideas and cheerful laughter. At the end of "
        "the day, they went home feeling proud, happy, and ready for another "
        "beautiful adventure tomorrow."
    )


def generate_story(
    caption: str,
    min_words: int = 50,
    max_words: int = 100,
) -> str:
    """
    Create a 50-100 word child-safe story based on the image caption.

    The function tries multiple generations. If none meets all criteria,
    it returns a pre-written safe fallback story instead of leaving the
    user with an empty result.
    """
    generator = load_story_generator()
    best_story = ""

    prompt = (
        "Write a gentle, happy, child-safe story for young children. "
        f"The picture shows: {caption}. "
        "Use simple English, friendly characters, and a positive ending. "
        "Story: "
    )

    for _ in range(3):
        result = generator(
            prompt,
            max_new_tokens=130,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        generated_text = result[0]["generated_text"]

        story = clean_story(
            text=generated_text,
            prompt=prompt,
            max_words=max_words,
        )

        if not story:
            continue

        if not is_child_safe(story):
            continue

        if len(story.split()) > len(best_story.split()):
            best_story = story

        if len(story.split()) >= min_words:
            return story

    if best_story:
        return best_story

    return fallback_story(caption)


def text_to_speech(text: str) -> bytes:
    """
    Convert text to MP3 bytes using Google Text-to-Speech.
    """
    if not text or not text.strip():
        raise ValueError("Cannot convert empty text to speech.")

    tts = gTTS(
        text=text,
        lang="en",
        slow=False,
    )

    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return audio_buffer.read()


# ---------- Streamlit user interface ----------
def main():
    st.title("📖 Magic Story Teller")
    st.write(
        "Upload a picture and I will describe it, write a child-safe story, "
        "and read the story aloud!"
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, and PNG.",
    )

    if uploaded_file is None:
        st.info("Please upload an image to begin.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("This file could not be read as an image. Please upload a JPG or PNG file.")
        return
    except Exception as error:
        st.error(f"Unable to open this image: {error}")
        return

    image.thumbnail((512, 512))
    st.image(image, caption="Your picture", use_container_width=True)

    try:
        with st.spinner("Looking at your picture..."):
            caption = generate_caption(image)

        st.info(f"**What I see:** {caption}")

    except Exception as error:
        st.error(
            "I could not analyse this image. "
            "Please try another image or reboot the app."
        )
        st.exception(error)
        return

    try:
        with st.spinner("Writing your story..."):
            story = generate_story(caption)

        if not story or not story.strip():
            st.warning(
                "Sorry, I could not write a story this time. "
                "Please try another image."
            )
            return

        st.success("Here is your story!")
        st.write(story)

    except Exception as error:
        st.error("I could not generate a story this time.")
        st.exception(error)
        return

    try:
        with st.spinner("Recording the story..."):
            audio_bytes = text_to_speech(story)

        st.audio(
            audio_bytes,
            format="audio/mp3",
        )

    except Exception as error:
        st.warning(
            "The story was created, but the audio could not be generated. "
            "Please check your internet connection and try again."
        )
        st.exception(error)


if __name__ == "__main__":
    main()
