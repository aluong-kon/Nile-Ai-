from pydantic import ValidationError
import pytest
from app import ChatIn, SpeakIn, TranslateIn


def test_chat_defaults_to_english():
    body = ChatIn(message="Hello")
    assert body.lang == "en"
    assert body.session_id is None


def test_chat_rejects_invalid_language():
    with pytest.raises(ValidationError):
        ChatIn(message="Hello", lang="fr")


def test_translate_rejects_invalid_direction():
    with pytest.raises(ValidationError):
        TranslateIn(text="Hello", direction="sideways")


def test_speak_rejects_empty_text():
    with pytest.raises(ValidationError):
        SpeakIn(text="")
