import ast
import asyncio
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from case_management import create_case, init_db, list_cases, get_case_by_id
from research_planning import parse_search_plan


class PlanningTests(unittest.TestCase):
    def plan(self):
        return {"queries": [
            {"query": query, "researchGoal": "Extract the reported evidence and verify its applicability to goods and services pair comparison.",
             "scope": "direct" if i < 3 else "supporting", "facet": facet}
            for i, (query, facet) in enumerate([
                ('goods services similarity Nice classification NLP', 'task'),
                ('trademark goods services text pair dataset', 'datasets'),
                ('goods services similarity classification evaluation F1', 'evaluation'),
                ('neuro symbolic text matching constraint learning', 'architecture'),
                ('legal ontology goods services complementarity substitutability', 'constraints'),
                ('text pair exact match synonym inclusion classification', 'features'),
            ])]}

    def test_preserves_queries_and_specific_goals(self):
        plan = self.plan()
        result = parse_search_plan(json.dumps(plan))
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0]['query'], plan['queries'][0]['query'])
        self.assertIn('[Supporting / architecture]', result[3]['researchGoal'])

    def test_rejects_overconstrained_query_and_empty_goal(self):
        for field, value in [('query', 'a AND b AND c AND d'), ('researchGoal', '')]:
            plan = self.plan()
            plan['queries'][0][field] = value
            with self.assertRaises(ValueError):
                parse_search_plan(json.dumps(plan))

    def test_deep_engine_keeps_all_approved_queries(self):
        path = Path('engines/gpt-researcher/gpt_researcher/skills/deep_research.py')
        tree = ast.parse(path.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'generate_search_queries')
        namespace = {'List': list, 'Dict': dict}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        approved = parse_search_plan(json.dumps(self.plan()))
        obj = SimpleNamespace(researcher=SimpleNamespace(preplanned_queries=approved))
        result = asyncio.run(namespace['generate_search_queries'](obj, 'brief', num_queries=3))
        self.assertEqual(result, approved)

    def test_missing_case_is_not_recreated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / 'state.sqlite'
            init_db(db)
            case = create_case(db, root, 'Study')
            self.assertEqual(len(list_cases(db)), 1)
            shutil.rmtree(case.root_dir)
            self.assertEqual(list_cases(db), [])
            self.assertFalse(Path(case.root_dir).exists())
            self.assertIsNotNone(get_case_by_id(db, case.id))
            Path(case.root_dir).mkdir()
            self.assertEqual(len(list_cases(db)), 1)


if __name__ == '__main__':
    unittest.main()
