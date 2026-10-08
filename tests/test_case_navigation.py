import ast
from pathlib import Path
import textwrap
import unittest
from streamlit.testing.v1 import AppTest


class NavigationTests(unittest.TestCase):
    def test_repeated_switches_and_navigation(self):
        source = Path('app.py').read_text()
        callbacks = '\n'.join(ast.unparse(n) for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name in {'set_navigation', 'switch_active_case'})
        start = source.index('    case_options = [case.slug')
        end = source.index('    workspace_title =', start)
        setup = """
import streamlit as st
from types import SimpleNamespace
def clear_research_plan_state(): st.session_state['research_plan_items'] = []
def reset_plan_widget_keys(): pass
all_cases = [SimpleNamespace(slug='alpha', name='Alpha'), SimpleNamespace(slug='beta', name='Beta')]
case_lookup = {c.slug: c for c in all_cases}
selected_case = case_lookup[st.session_state.get('selected_case_slug', 'alpha')]
"""
        app = AppTest.from_string(setup + callbacks + '\n' + textwrap.dedent(source[start:end]) + """
st.button('Home', on_click=set_navigation, args=('home',))
if st.button('Logout'):
    st.session_state['logged_out'] = True
""").run()
        for slug in ['beta', 'alpha', 'beta', 'alpha']:
            app.selectbox(key='case_picker').select(slug).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['selected_case_slug'], slug)
            self.assertEqual(app.selectbox(key='case_picker').value, slug)
        app.radio(key='navigation_view').set_value('workspace').run()
        next(b for b in app.button if b.label=='Home').click().run()
        self.assertEqual(app.session_state['app_view'], 'home')
        next(b for b in app.button if b.label=='Logout').click().run()
        self.assertTrue(app.session_state['logged_out'])


if __name__ == '__main__': unittest.main()
