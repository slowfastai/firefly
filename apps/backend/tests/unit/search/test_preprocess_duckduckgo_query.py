import pytest

from search._duckduckgo_search import preprocess_duckduckgo_query


def test_removes_wildcard_after_site():
    q = "climate site:*.gov data"
    out = preprocess_duckduckgo_query(q)
    assert "site:*.gov" not in out
    assert "site:gov" in out


def test_dedup_subdomain_when_parent_present():
    q = "foo site:azure.microsoft.com bar site:microsoft.com baz"
    out = preprocess_duckduckgo_query(q)
    # Keep only the parent
    assert "site:microsoft.com" in out
    # Remove subdomain duplicate
    assert "azure.microsoft.com" not in out
    # Parent should appear exactly once
    assert out.count("site:microsoft.com") == 1


def test_preserve_negative_site_filters():
    q = "site:example.com -site:sub.example.com info"
    out = preprocess_duckduckgo_query(q)
    # Positive kept (no parent/subdomain conflict)
    assert "site:example.com" in out
    # Negative untouched
    assert "-site:sub.example.com" in out


def test_dedup_duplicate_same_domain_case_insensitive():
    q = "site:EXAMPLE.com docs site:example.com"
    out = preprocess_duckduckgo_query(q)
    # Only one positive site filter remains
    assert out.lower().count("site:example.com") == 1


def test_normalizes_for_comparison_but_keeps_parent_token():
    # Contains a URL-like site and its parent; we should keep the parent token and drop the URL-like one
    q = "site:https://www.example.com/path?q=1 site:example.com other"
    out = preprocess_duckduckgo_query(q)
    assert "site:example.com" in out
    assert "site:https://www.example.com/path" not in out


def test_multi_level_hierarchy_keeps_only_root():
    q = "site:a.b.example.com site:b.example.com site:example.com"
    out = preprocess_duckduckgo_query(q)
    assert "site:example.com" in out
    assert "a.b.example.com" not in out
    assert "b.example.com" not in out


def test_no_site_filters_returns_original():
    q = "how the sun formed nebular hypothesis"
    out = preprocess_duckduckgo_query(q)
    assert out == q


def test_remove_positive_filetype_html_and_ext_htm():
    q = "foo filetype:html bar ext:htm baz"
    out = preprocess_duckduckgo_query(q)
    assert "filetype:html" not in out.lower()
    assert "ext:htm" not in out.lower()
    # Core terms remain
    assert "foo" in out and "bar" in out and "baz" in out


def test_preserve_negative_filetype_html():
    q = "-filetype:html site:example.com"
    out = preprocess_duckduckgo_query(q)
    assert "-filetype:html" in out.lower()
    assert "site:example.com" in out.lower()


def test_boolean_cleanup_after_removal():
    q = "OR filetype:html foo AND"
    out = preprocess_duckduckgo_query(q)
    # After removing positive filetype:html, stray OR/AND at edges are cleaned
    assert out == "foo"


def test_keep_other_filetypes():
    q = "filetype:pdf site:example.com"
    out = preprocess_duckduckgo_query(q)
    assert "filetype:pdf" in out.lower()
    assert "site:example.com" in out.lower()


def test_ext_variants_case_insensitive():
    q = "EXT:HTML FiLeTyPe:HtM keep"
    out = preprocess_duckduckgo_query(q)
    assert "ext:html" not in out.lower()
    assert "filetype:htm" not in out.lower()
    assert "keep" in out
