import json
import unittest
from types import SimpleNamespace
from chat_policy import evaluate_question, OFF_TOPIC, UNAVAILABLE
from chat_formatting import answer_blocks


class PolicyTests(unittest.TestCase):
    def test_decisions_and_invalid_output(self):
        case = SimpleNamespace(name='Text segmentation', description='NLP algorithms', research_goal='Compare segmenters')
        for raw, allowed, message in [('{"in_scope": false}', False, OFF_TOPIC),
                                     ('{"in_scope": true}', True, ''),
                                     ('{"in_scope": "true"}', False, UNAVAILABLE),
                                     ('invalid', False, UNAVAILABLE)]:
            result = evaluate_question('Who is the best footballer?', case, [], lambda *a, **k: raw, {})
            self.assertEqual(result, (allowed, message))

    def test_classifier_sees_case_and_followup_context(self):
        case = SimpleNamespace(name='Algorithms', description='Text segmentation', research_goal='Implement TextTiling')
        def complete(messages, settings, **kwargs):
            data = json.loads(messages[1]['content'])
            self.assertEqual(data['study']['research_goal'], 'Implement TextTiling')
            self.assertEqual(data['history'][0]['content'], 'TextTiling')
            return '{"in_scope": true}'
        self.assertTrue(evaluate_question('Implement it in Python', case,
                        [{'role':'assistant','content':'TextTiling'}], complete, {})[0])

    def test_model_failure_is_closed(self):
        def fail(*args, **kwargs): raise TimeoutError()
        self.assertFalse(evaluate_question('Question', SimpleNamespace(name='Study'), [], fail, {})[0])

    def test_code_fences_and_prose(self):
        text = 'An example:\n\n```py\nprint("hello")\n```\n\n```unknownlang\nx\n```\nDone.'
        blocks = list(answer_blocks(text))
        code = [b for b in blocks if b[0]=='code']
        self.assertEqual([b[2] for b in code], ['python', 'text'])
        self.assertEqual(code[0][1], 'print("hello")\n')
        self.assertTrue(blocks[-1][1].endswith('Done.'))


if __name__ == '__main__': unittest.main()
