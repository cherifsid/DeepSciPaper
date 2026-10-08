"""Parse fenced code with a Markdown parser, preserving surrounding prose."""
from markdown_it import MarkdownIt
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound


def answer_blocks(markdown):
    lines = markdown.splitlines(keepends=True)
    cursor = 0
    for token in MarkdownIt().parse(markdown):
        if token.type not in {'fence', 'code_block'} or token.map is None:
            continue
        start, end = token.map
        if start > cursor:
            yield 'markdown', ''.join(lines[cursor:start]), None
        language = token.info.strip().split()[0] if token.info.strip() else 'text'
        aliases = {'py': 'python', 'js': 'javascript', 'ts': 'typescript', 'sh': 'bash', 'c++': 'cpp'}
        language = aliases.get(language.lower(), language.lower())
        try:
            get_lexer_by_name(language)
        except ClassNotFound:
            language = 'text'
        yield 'code', token.content, language
        cursor = end
    if cursor < len(lines):
        yield 'markdown', ''.join(lines[cursor:]), None
