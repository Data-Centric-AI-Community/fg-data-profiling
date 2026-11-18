"""
Test for issue 1499:
https://github.com/Data-Centric-AI-Community/fg-data-profiling/issues/1499

Repository hygiene: ship a SECURITY.md describing a private channel for
vulnerability reports, as recommended by GitHub's repo-security
documentation
(https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository).

The test below would fail on origin/develop (no SECURITY.md present) and
passes on the fix branch. It also asserts the file mentions the keys a
researcher needs in order to actually use the policy: a private reporting
channel, a reply-time expectation, and a "do not file public issues"
warning.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_security_md_exists_at_repo_root():
    """GitHub auto-detects SECURITY.md in the repo root, /docs, or /.github."""
    candidates = [
        REPO_ROOT / "SECURITY.md",
        REPO_ROOT / ".github" / "SECURITY.md",
        REPO_ROOT / "docs" / "SECURITY.md",
    ]
    found = [p for p in candidates if p.is_file()]
    assert found, "Expected a SECURITY.md at one of: " + ", ".join(
        str(p.relative_to(REPO_ROOT)) for p in candidates
    )


def test_security_md_describes_a_private_reporting_channel():
    path = next(
        p
        for p in [
            REPO_ROOT / "SECURITY.md",
            REPO_ROOT / ".github" / "SECURITY.md",
            REPO_ROOT / "docs" / "SECURITY.md",
        ]
        if p.is_file()
    )
    text = path.read_text(encoding="utf-8").lower()
    # The three things a researcher needs to know:
    #   1. Where to report privately (advisory or email).
    #   2. Not to file a public issue.
    #   3. What response window to expect.
    assert (
        "security advisory" in text
        or "security advisories" in text
        or "@" in text  # an email address — the minimal alternative
    ), "SECURITY.md must point to a private reporting channel (advisory or email)."
    assert (
        "public" in text and "issue" in text
    ), "SECURITY.md must warn against filing public issues for vulnerabilities."
    assert any(
        kw in text for kw in ("days", "business day", "response", "reply")
    ), "SECURITY.md must give an expected response window."
