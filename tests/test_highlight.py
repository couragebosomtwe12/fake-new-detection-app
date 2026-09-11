from app.explain import TokenWeight
from app.highlight import build_highlighted_html


def test_build_highlighted_html_wraps_matches_and_escapes_html():
    tokens = [TokenWeight(token="miracle", weight=0.8), TokenWeight(token="council", weight=-0.4)]
    result = build_highlighted_html("A miracle cure from the council <script>", tokens)
    assert "<mark" in result
    assert "&lt;script&gt;" in result
    assert "<script>" not in result


def test_build_highlighted_html_returns_escaped_text_when_no_tokens():
    assert build_highlighted_html("<b>hi</b>", []) == "&lt;b&gt;hi&lt;/b&gt;"


def test_build_highlighted_html_matches_case_insensitively_on_word_boundaries():
    tokens = [TokenWeight(token="Miracle", weight=0.6)]
    result = build_highlighted_html("MIRACLE cures and miraculous claims", tokens)
    assert result.count("<mark") == 1  # "miraculous" must not match "miracle"
