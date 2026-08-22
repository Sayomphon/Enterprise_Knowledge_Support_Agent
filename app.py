"""Streamlit entry point: compose the two pages and run the selected one.

The interface itself lives in the ``ui`` package -- strings in
``ui.labels``, stylesheets in ``ui.styles``, pure formatting in
``ui.formatting``, the cached pipeline in ``ui.runtime``, and the two
surfaces in ``ui.assistant`` and ``ui.console``. This file holds what
Streamlit needs at import time and nothing else, so every part of the
interface below it can be imported by a test; ``app.py`` itself still
calls ``main()`` at module scope, because that is how Streamlit runs a
script.

The app serves two pages: the employee assistant on ``/`` and the audit
console on ``/console``. Each owns its rail, app bar, and layout measure
so they read as separate products, and the console holds every technical
detail (reason codes, scores, latency, runtime state). That separation is
information architecture for the demo only; it is NOT a security boundary
and no authentication or RBAC is implemented.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from ui.assistant import _employee_page
from ui.console import _console_page
from ui.labels import APP_TITLE, EMPLOYEE_VIEW, OPS_VIEW
from ui.runtime import _PAGE_REFS
from ui.styles import _DESIGN_SYSTEM_CSS


def main() -> None:
    """Compose the two separated pages and run the selected one.

    The assistant answers on ``/`` and the console on ``/console``. Each
    page carries its own rail, header, and layout tokens so the two read as
    different products. This is information architecture, not access
    control: no authentication or RBAC exists, and both rails say so.
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    load_dotenv()
    st.markdown(_DESIGN_SYSTEM_CSS, unsafe_allow_html=True)
    _PAGE_REFS.update(
        {
            # The default page always answers on the root URL, so it takes
            # no url_path of its own: "/" is the assistant, "/console" is
            # the audit console.
            "assistant": st.Page(
                _employee_page,
                title=EMPLOYEE_VIEW,
                default=True,
            ),
            "console": st.Page(
                _console_page, title=OPS_VIEW, url_path="console"
            ),
        }
    )
    st.navigation(list(_PAGE_REFS.values()), position="hidden").run()


main()
