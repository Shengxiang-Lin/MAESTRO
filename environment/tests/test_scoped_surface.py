"""A scoped certificate binds the real sentence text to the frozen query."""
import unittest
from unittest.mock import patch
from environment.tests import test_stage2_surface_certification as fixtures
from environment.question_generation.backend import LLMQuestionAuthor, OpenAICompatibleBackend, PassthroughCriticBackend
from environment.question_generation.pipeline import _realize_one
from environment.question_generation.scoped_surface import build_scoped_plan, read_scoped, read_scoped_semantics, render_scoped
from environment.question_generation.verification import certify_scoped_surface, validate_surface_certificate
from environment.trajectory_simulation.compiler import compile_gold_solution
from environment.trajectory_simulation.eligibility import controlled_question_reference

class ScopedSurfaceTests(unittest.TestCase):
    def fixture(self):
        payload, world, capability, rendered = fixtures.Stage2SurfaceCertificationTests().data()
        realized = {**payload, 'question': rendered.question, 'capability_world': capability}
        gold = compile_gold_solution(realized, world, rendered, schedule='compact')
        groups = build_scoped_plan(world, rendered.question, gold,
                                   controlled_question_reference(gold, world, rendered))
        return payload, world, capability, rendered, gold, groups

    def test_scoped_text_publishes_after_one_style_call(self):
        payload, world, capability, rendered, _, groups = self.fixture()
        author = LLMQuestionAuthor(OpenAICompatibleBackend('test://local', 'test'))
        with patch.object(OpenAICompatibleBackend, 'complete_json',
                          return_value={'choices':[0]*len(groups)}) as call:
            ok, row, _ = _realize_one(payload, world, capability, author,
                                      PassthroughCriticBackend(), question_max_attempts=2,
                                      use_critic=True, surface_mode='scoped')
        self.assertTrue(ok)
        self.assertEqual(row['publication_status'], 'naturalized_certified')
        self.assertEqual(call.call_count, 1)
        self.assertTrue(validate_surface_certificate(row, world, rendered.question)['eligible'])

    def test_consumer_reparses_even_if_changed_text_has_fresh_hashes(self):
        payload, world, capability, rendered, gold, groups = self.fixture()
        question = render_scoped(groups, [0]*len(groups)) + ' Exclude Alice.'
        original = render_scoped(groups, [0]*len(groups))
        cert, review = certify_scoped_surface(payload['task_id'], world, rendered.question, original, groups)
        from environment.trajectory_simulation.eligibility import digest
        cert['question_sha256'] = review['question_sha256'] = digest(question)
        cert['review_sha256'] = digest(review)
        realized = {**payload, 'question':question, 'capability_world':capability,
                    'surface_certificate':cert, 'authoring':{'surface_review':review}}
        with self.assertRaisesRegex(ValueError, 'round-trip mismatch'):
            validate_surface_certificate(realized, world, rendered.question, gold=gold, rendered=rendered)

    def test_scope_values_join_and_output_must_survive_readback(self):
        def node(i, op, deps=(), args=()):
            return {'node_id':i, 'op':op, 'dependencies':list(deps), 'inputs':list(args)}
        world = {'program':{'nodes':[
            node(0,'FIND_ALL'),
            node(1,'FILTER_DATE',(0,),('schema:deathDate','=','1997-08-01')),
            node(2,'RELATE',(1,),('schema:deathPlace','forward')),
            node(3,'FILTER_DATE',(2,),('schema:endDate','=','2004-05-03')),
            node(4,'RELATE',(3,),('schema:deathPlace','backward')),
            node(5,'FILTER_TYPE',(4,),('schema:Person',)),
            node(6,'FIND_ALL'),
            node(7,'FILTER_STRING',(6,),('schema:telephoneCountryCode','=','+650')),
            node(8,'FILTER_NUMBER',(7,),('schema:malePopulation','=','488')),
            node(9,'INTERSECT',(5,8)), node(10,'QUERY_NAME',(9,)),
        ]}}
        canonical = 'Person death date 1997-08-01 death place end date 2004-05-03 telephone country code +650 male population 488'
        groups = build_scoped_plan(world, canonical, {'fragments':[]}, {'verified_identity_clues':[]})
        question = render_scoped(groups, [0]*len(groups))
        self.assertIn('August 1, 1997', question)
        self.assertIn('May 3, 2004', question)
        self.assertEqual(read_scoped(question, groups), [0]*len(groups))
        for changed in (
            question.replace('August 1, 1997', 'temporary').replace('May 3, 2004', 'August 1, 1997').replace('temporary', 'May 3, 2004'),
            question.replace('recorded as places of death for', 'recorded as dying at'),
            question.replace(' and a male population of 488', ''),
            question.replace('+650', '650'),
            question.replace('belong to both', 'belong to either'),
            question.replace('Which people', 'How many people'),
        ):
            self.assertNotEqual(changed, question)
            self.assertIsNone(read_scoped(changed, groups), changed)

    def test_every_approved_wording_reads_the_same_typed_scoped_query(self):
        _, _, _, _, _, groups = self.fixture()
        for i, options in enumerate(groups):
            for j in range(len(options)):
                choices = [0]*len(groups)
                choices[i] = j
                text = render_scoped(groups, choices)
                result = read_scoped_semantics(text, groups)
                self.assertEqual(result['query'], groups.query)
                self.assertEqual(' '.join(s['text'] for s in result['sentences']), text)
        with self.assertRaisesRegex(ValueError, 'compiler-derived'):
            read_scoped_semantics('Which people?', [('Which people?',)])

    def test_added_exclusion_or_removed_sentence_is_not_accepted(self):
        _, _, _, _, _, groups = self.fixture()
        text = render_scoped(groups, [0]*len(groups))
        for changed in (text + ' Exclude the anchor person.',
                        text.replace('Which', 'Which other', 1),
                        ' '.join(groups[-1][:1])):
            if changed != text:
                self.assertIsNone(read_scoped_semantics(changed, groups))

    def test_consumer_rejects_forged_semantics_even_with_valid_text_and_hashes(self):
        import copy
        from environment.trajectory_simulation.eligibility import digest
        payload, world, capability, rendered, gold, groups = self.fixture()
        question = render_scoped(groups, [0]*len(groups))
        cert, review = certify_scoped_surface(payload['task_id'], world, rendered.question, question, groups)
        review = copy.deepcopy(review)
        review['semantic_readback']['query']['definition'] = 'COUNT'
        cert['review_sha256'] = digest(review)
        realized = {**payload, 'question': question, 'capability_world': capability,
                    'surface_certificate': cert, 'authoring': {'surface_review': review}}
        with self.assertRaisesRegex(ValueError, 'round-trip mismatch'):
            validate_surface_certificate(realized, world, rendered.question, gold=gold, rendered=rendered)
