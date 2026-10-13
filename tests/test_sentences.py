"""Sentence grouping from word timestamps."""
from __future__ import annotations

from app.transcribe import Segment, Word, sentences_from_words


def test_words_group_into_sentences_with_first_and_last_word_times() -> None:
    words = [
        Word(0.10, 0.40, " I"), Word(0.45, 0.80, " went"), Word(0.85, 1.20, " home."),
        Word(1.60, 1.90, " It"), Word(1.95, 2.30, " was"), Word(2.35, 2.70, " late!"),
        Word(3.00, 3.40, " Really?"),
        Word(3.80, 4.10, " Trailing"), Word(4.15, 4.50, " words"),
    ]
    assert sentences_from_words(words) == [
        Segment(0.10, 1.20, "I went home."),
        Segment(1.60, 2.70, "It was late!"),
        Segment(3.00, 3.40, "Really?"),
        Segment(3.80, 4.50, "Trailing words"),
    ]


def test_non_latin_sentence_enders() -> None:
    words = [Word(0.0, 0.5, "یہ"), Word(0.5, 1.0, "ہے۔"), Word(1.2, 1.5, "कहाँ"), Word(1.5, 2.0, "है।")]
    assert [s.text for s in sentences_from_words(words)] == ["یہ ہے۔", "कहाँ है।"]


def test_blank_words_are_ignored() -> None:
    assert sentences_from_words([Word(0, 1, " "), Word(1, 2, "ok.")]) == [Segment(1, 2, "ok.")]
