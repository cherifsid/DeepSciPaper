"""Exercise the production chat view with deterministic, offline retrieval."""
from pathlib import Path
import textwrap
import unittest

from streamlit.testing.v1 import AppTest


def preview_source():
    source = Path('app.py').read_text()
    start = source.index('    if "chat_messages" not in st.session_state:')
    setup = r"""
import streamlit as st
from chat_formatting import answer_blocks
from pathlib import Path
from types import SimpleNamespace
settings = {'model': 'test', 'backend': 'Local Ollama'}
paths = SimpleNamespace(bib_pdf=Path('/tmp'))
selected_case = SimpleNamespace(id=1)
APP_DB_PATH = multimodal_store_path = graph_store_path = Path('/tmp')
def persist_chat_message(*args, **kwargs): pass
def maybe_auto_sync_case(*args): pass
def clear_case_chat_history(*args): pass
def runtime_preflight(*args, **kwargs): return []
def mount_chat_composer(): pass
def evaluate_question(*args): return True, ''
def llm_complete(*args, **kwargs): return ''
def render_markdown_block(text): st.markdown(text)
def render_evidence_resources(*args, **kwargs): st.write('Paper A, Methods, page 3')
def render_artifacts(*args, **kwargs): pass
def multimodal_chat_answer(*args, **kwargs):
    return {'answer_markdown': '## Findings\n\n| Method | F1 |\n|---|---|\n| A | 0.82 |',
            'evidence_references': [{'evidence_id': 'E1'}]}
graph_chat_answer = multimodal_chat_answer
"""
    return setup + textwrap.dedent(source[start:])


class ChatUITests(unittest.TestCase):
    def test_off_topic_never_reaches_retrieval(self):
        source = preview_source().replace("def evaluate_question(*args): return True, ''",
                                         "def evaluate_question(*args): return False, 'Outside this study.'")
        app = AppTest.from_string(source).run()
        app.chat_input[0].set_value('Who is the best footballer?').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['chat_messages'][-1]['content'], 'Outside this study.')
        self.assertEqual(app.session_state['chat_messages'][-1]['evidence'], [])

    def test_repeated_submissions_and_modes(self):
        app = AppTest.from_string(preview_source()).run()
        app.chat_input[0].set_value('Compare methods').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 2)
        app.selectbox(key='rag_answer_mode').select('research').run()
        app.chat_input[0].set_value('Find conflicting evidence').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 4)
        self.assertEqual(app.session_state['chat_messages'][-1]['mode'], 'research')
        self.assertEqual(len(app.chat_input), 1)

    def test_clear_requires_confirmation(self):
        app = AppTest.from_string(preview_source()).run()
        app.chat_input[0].set_value('Summarize').run()
        self.assertFalse(any(b.label == 'Clear conversation' for b in app.button))
        app.checkbox(key='confirm_clear_chat').check().run()
        next(b for b in app.button if b.label == 'Clear conversation').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 0)


if __name__ == '__main__':
    unittest.main()
