"""Theme injection for the ClaimLens UI.

The stylesheet lives in ``ui/assets/theme.css`` and is injected once per script run as a
style-only ``st.html`` block (Streamlit routes style-only HTML to the event container, so it
takes no space in the layout). Fonts use local system stacks first; the optional Google Fonts
import sits in its own ``<style>`` element so that, offline, a failed request cannot delay the
main stylesheet - the app simply falls back to Segoe UI / Consolas.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
CSS_FILE = ASSETS / "theme.css"
FONT_IMPORT = ("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
               "&family=JetBrains+Mono:wght@400;500;600&display=swap');")

# Live character counter for the claim box (form widgets only report their value on submit).
# Static code, no user data; it only reads the textarea length and updates #cl-counter.
COUNTER_JS = """
<script>
(function () {
  if (window.__clCounterBound) { return; }
  window.__clCounterBound = true;
  document.addEventListener('input', function (e) {
    var t = e.target;
    if (!t || t.tagName !== 'TEXTAREA' || t.getAttribute('aria-label') !== 'Scientific claim') { return; }
    var c = document.getElementById('cl-counter');
    if (!c) { return; }
    var n = t.value.length, max = parseInt(c.getAttribute('data-max') || '2000', 10);
    var span = c.querySelector('.cl-counter-n');
    if (span) { span.textContent = n.toLocaleString('en-US'); }
    if (n > max) { c.classList.add('cl-over'); } else { c.classList.remove('cl-over'); }
  }, true);
})();
</script>
"""


def stylesheet() -> str:
    """Read on every run (a few KB) so edits to the CSS show up on the next rerun."""
    try:
        return CSS_FILE.read_text(encoding="utf-8")
    except OSError:  # a missing stylesheet must not break the app; it only loses styling
        return ""


def inject_theme() -> None:
    st.html(f"<style>{FONT_IMPORT}</style>")
    css = stylesheet()
    if css:
        st.html(f"<style>{css}</style>")
