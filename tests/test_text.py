from finbert.text import split_sentences


def test_splits_on_sentence_end():
    assert split_sentences("Revenue rose. Costs fell! Is that good? Yes.") == [
        "Revenue rose.",
        "Costs fell!",
        "Is that good?",
        "Yes.",
    ]


def test_keeps_abbreviations_and_initials_together():
    text = "Apple Inc. reported results in the U.S. market. J. Smith said margins held."
    assert split_sentences(text) == [
        "Apple Inc. reported results in the U.S. market.",
        "J. Smith said margins held.",
    ]


def test_does_not_split_decimals_or_lowercase_continuation():
    assert split_sentences("EPS was 1.25 vs. 1.10 a year ago. Guidance unchanged.") == [
        "EPS was 1.25 vs. 1.10 a year ago.",
        "Guidance unchanged.",
    ]


def test_paragraphs_and_whitespace():
    assert split_sentences("  First line\n\nSecond   paragraph\nwraps here  ") == [
        "First line",
        "Second paragraph wraps here",
    ]
    assert split_sentences("   \n\n ") == []


def test_splits_after_amount_units():
    # "mn"/"bn" usually end a sentence in financial text ("rose to EUR 13.1 mn.") and must not block a split.
    assert split_sentences("Profit rose to EUR 13.1 mn. Sales were USD 2 bn. Net loss widened.") == [
        "Profit rose to EUR 13.1 mn.",
        "Sales were USD 2 bn.",
        "Net loss widened.",
    ]
