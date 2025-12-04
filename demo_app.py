"""Streamlit demo app for emotion-aware responses backed by trained models."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import openai
import pandas as pd
import streamlit as st
import torch

from config import Config
from src.feature_extractor import TextFeatureExtractor
from src.models import MultimodalERC
from src.preprocessor import TextPreprocessor

EMOTION_LABELS: List[str] = Config.EMOTION_LABELS
EMOTION_TO_ID: Dict[str, int] = Config.EMOTION_MAP
ID_TO_EMOTION: Dict[int, str] = {idx: emotion for emotion, idx in EMOTION_TO_ID.items()}

MODEL_REGISTRY = {
    "CAHME": {
        "path": Path("best_model_cahme.pth"),
        "architecture": "cahme",
        "help": "Fusion model with gated cross-modal attention."
    },
    "M3F-Net": {
        "path": Path("best_model_m3fnet.pth"),
        "architecture": "m3fnet",
        "help": "Multi-scale temporal encoder with dynamic feature gating."
    },
}


@st.cache_resource(show_spinner=False)
def load_text_extractor() -> TextFeatureExtractor:
    """Lazy-load the DistilBERT text encoder used during training."""
    return TextFeatureExtractor()


@st.cache_resource(show_spinner=True)
def load_trained_model(label: str) -> Tuple[MultimodalERC, str]:
    """Load a trained checkpoint for inference."""
    registry_entry = MODEL_REGISTRY[label]
    checkpoint_path = registry_entry["path"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Please place the trained file in the project root."
        )

    config = Config()
    architecture = registry_entry["architecture"]
    model = MultimodalERC(
        config,
        architecture=architecture,
        num_classes=len(config.EMOTION_LABELS),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, architecture


def predict_emotion(user_text: str, model_label: str) -> Tuple[str, Dict[str, float]]:
    """Infer the dominant emotion using a trained model."""
    model, architecture = load_trained_model(model_label)
    extractor = load_text_extractor()

    cleaned = TextPreprocessor.clean_text(user_text)
    text_features = extractor.get_utterance_embedding(cleaned)
    text_tensor = torch.tensor(text_features).float().unsqueeze(0)
    audio_tensor = torch.zeros((1, Config.AUDIO_FEATURE_DIM))
    multimodal_tensor = torch.cat([text_tensor, audio_tensor], dim=1)

    with torch.no_grad():
        if architecture == "m3fnet":
            outputs = model(multimodal_features=multimodal_tensor.unsqueeze(1))
        else:
            current_features = {
                "text": text_tensor,
                "audio": audio_tensor,
                "multimodal": multimodal_tensor,
            }
            outputs = model(current_features=current_features, history_features=None)

        probabilities = torch.softmax(outputs["logits"], dim=1).squeeze(0)
        confidence_map = {
            ID_TO_EMOTION[idx]: round(prob.item(), 4)
            for idx, prob in enumerate(probabilities)
        }

        predicted_idx = int(torch.argmax(probabilities))
        predicted_emotion = ID_TO_EMOTION[predicted_idx]

    return predicted_emotion, confidence_map


def build_prompt(user_text: str, emotion: str) -> str:
    """Construct a concise prompt that pairs the text with the detected emotion."""
    return (
        "You are an empathetic assistant that tailors responses based on emotional cues. "
        f"The detected dominant emotion is '{emotion}'. "
        "Acknowledge the feeling, keep the tone supportive, and provide an appropriate follow-up."
        "You shouldn't ask a question in response in your follow-up."
        "You may exagerate the emotions in your response to make user feel more understood."
        f"User input: {user_text}"
    )


def generate_openai_response(user_text: str, emotion: str) -> str:
    """Send the emotion-tagged prompt to the OpenAI API and return the model's reply."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "Set the OPENAI_API_KEY environment variable to enable model responses. "
            "Using the detected emotion locally for now."
        )

    client = openai.OpenAI(api_key=api_key)
    prompt = build_prompt(user_text, emotion)
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate concise, empathetic replies that acknowledge the user's emotion "
                        "and keep the conversation moving forward."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=180,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        return f"OpenAI response unavailable: {exc}"


def main() -> None:
    st.set_page_config(page_title="EACA Demo", page_icon="😵‍💫", layout="centered")
    selected_model = st.selectbox(
        "Choose a trained model",
        list(MODEL_REGISTRY.keys()),
    )

    user_text = st.text_area("Your message", height=150)
    submitted = st.button("Analyze emotion", type="primary")

    if submitted:
        if not user_text.strip():
            st.warning("Please enter a message first.")
            return

        with st.spinner("Running trained model..."):
            try:
                dominant_emotion, scores = predict_emotion(user_text, selected_model)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                st.error(f"Model inference failed: {exc}")
                return

        st.success(f"Detected emotion: **{dominant_emotion.title()}**")
        st.markdown("### OpenAI response")
        response = generate_openai_response(user_text, dominant_emotion)
        st.write(response)

        st.markdown("### Emotion probabilities")
        score_df = (
            pd.DataFrame.from_dict(scores, orient="index", columns=["probability"])
            .reindex(EMOTION_LABELS)
        )
        st.bar_chart(score_df, height=240)

        with st.expander("Preprocessing details"):
            cleaned = TextPreprocessor.clean_text(user_text)
            tokens = TextPreprocessor.tokenize(cleaned)
            st.write(f"Cleaned text: {cleaned}")
            st.write(f"Tokens ({len(tokens)}): {tokens}")


if __name__ == "__main__":
    main()
