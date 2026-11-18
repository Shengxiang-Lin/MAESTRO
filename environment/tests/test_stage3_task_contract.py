"""Semantic handoff, repair and fallback regressions independent of live LLMs."""
import json
import unittest
from environment.base_orchestration.task_contract import compile_task_contract, contract_prose_issues
from environment.base_orchestration.action_author import (
    author_delegate_action, _stage3_compile_task_prose, proposed_template_action,
    _critic_dispatch_rewrite_request,
)


def assignment():
    target = {'external_inputs': ['upstream_private_node'],
              'input_schema': {'input_0': ['project_id']},
              'source_ids': ['event_ledger'], 'outputs': {'typed_relation': ['event_id']},
              'operator_chain': [
                  {'id': 'private_a', 'op': 'lookup_by_key', 'inputs': ['upstream_private_node'],
                   'source_id': 'event_ledger', 'input_key_field': 'project_id', 'source_key_field': 'project_id'},
                  {'id': 'private_b', 'op': 'filter_gte', 'inputs': ['private_a'], 'field': 'event_year', 'value': 2025},
                  {'id': 'private_c', 'op': 'project', 'inputs': ['private_b'], 'fields': ['event_id']}]}
    contract = compile_task_contract(target, {'event_ledger': {'title': 'Project event ledger'}})
    return {'target_instruction': contract['reference_instruction'], 'semantic_contract': contract,
            'context': 'Previously returned result: PRJ-ABC123; Sub-Agent summary: The project was identified.',
            'model': 'model_1', 'tools': [], 'source_ids': ['event_ledger'],
            'public_source_descriptions': [{'source_id': 'event_ledger', 'title': 'Project event ledger', 'modality': 'table'}],
            'output_fields': ['event_id'], 'needs_upstream': True, 'keyed_lookup': True}


class Author:
    def __init__(self, bad=False): self.requests = []; self.bad = bad
    def complete_json(self, system, payload):
        req = json.loads(payload); self.requests.append(req)
        return {'reasoning': 'The returned project identifier allows us to locate the eligible event records.',
                'tasks': [{'task_instruction': 'Inspect event_ledger and return all event_id records.' if self.bad else assignment()['target_instruction'],
                           'context': 'Use the identified project to retrieve the eligible events.'}]}


class Critic:
    def __init__(self, always_fail=False): self.calls = 0; self.always_fail = always_fail
    def complete_json(self, system, payload):
        self.calls += 1
        if self.calls == 1 or self.always_fail:
            return {'verdict': 'fail', 'grounded': False, 'assignment_fidelity': False, 'o1_style': True,
                    'reason': 'A local condition requires attention.', 'repair_hint': 'Do not replace the event filter.',
                    'issues': [{'task_index': 0, 'field': 'task_instruction', 'kind': 'missing_condition',
                                'quote': '', 'contract_step': 2}]}
        return {'verdict': 'pass', 'grounded': True, 'assignment_fidelity': True, 'o1_style': True}


class TaskContractTests(unittest.TestCase):
    def test_contract_preserves_filter_scope_and_excludes_executable_ids(self):
        c = assignment()['semantic_contract']; text = json.dumps(c)
        self.assertNotIn('private_a', text); self.assertNotIn('upstream_private_node', text)
        self.assertEqual(c['filters'][0]['scope'], 'event_ledger records')
        self.assertIn('event_year at least 2025', c['reference_instruction'])
        self.assertEqual(contract_prose_issues('Inspect event_ledger and return event_id.', c), ['condition_missing:2'])
        self.assertEqual(contract_prose_issues(c['reference_instruction'], c), [])

    def test_critic_repair_reaches_next_author_with_prior_draft(self):
        a, c = Author(), Critic()
        action, status = author_delegate_action(assignments=[assignment()], prompt='Find the eligible event for project PRJ-ABC123.',
            author=a, critic=c, max_revisions=1, freeze_assignment_prose=True,
            reasoning_only_stage3=True, naturalize_assignment_prose=True)
        self.assertEqual(status['mode'], 'llm_reviewed_contract_frozen')
        self.assertTrue(a.requests[1]['previous_draft'])
        self.assertTrue(any('event_year' in x and '2025' in x for x in a.requests[1]['repair_guidance']))
        self.assertEqual(len(status['failures']), 1)
        self.assertIn('candidate', status['failures'][0])
        self.assertEqual(status['critic_calls'], 2)

    def test_rejected_prose_fails_by_default_without_delegate_fallback(self):
        action, status = author_delegate_action(
            assignments=[assignment()],
            prompt='Find the eligible event for project PRJ-ABC123.',
            author=Author(bad=True), critic=Critic(), max_revisions=0,
            freeze_assignment_prose=True, reasoning_only_stage3=True,
            naturalize_assignment_prose=True,
        )
        self.assertIsNone(action)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['mode'], 'llm_reviewed_contract_frozen')
        self.assertTrue(status['failures'])
        self.assertFalse(any(
            failure.get('stage') == 'delegate_fallback_guard'
            for failure in status['failures']
        ))

    def test_legacy_delegate_fallback_is_explicit_opt_in_only(self):
        action, status = author_delegate_action(
            assignments=[assignment()],
            prompt='Find the eligible event for project PRJ-ABC123.',
            author=Author(bad=True), critic=Critic(), max_revisions=0,
            freeze_assignment_prose=True, reasoning_only_stage3=True,
            naturalize_assignment_prose=True, allow_delegate_fallback=True,
        )
        self.assertEqual(status['mode'], 'contract_frozen_delegate_fallback')
        self.assertFalse(status['reviewed'])
        self.assertIn('event_year at least 2025', action['params']['tasks'][0]['task_instruction'])
        self.assertNotIn('future_answer', json.dumps(action))

    def test_required_field_removal_triggers_scope_review(self):
        self.assertTrue(_critic_dispatch_rewrite_request({'verdict': 'fail', 'reason': 'The contract requires entity_final.',
                                                        'repair_hint': 'Remove entity_final from the list of fields to return.'}))

    def test_context_evidence_is_attached_once(self):
        a = assignment(); draft = {'reasoning': 'The available project identifier supports this investigation.',
            'tasks': [{'task_instruction': a['target_instruction'], 'context': a['context']}]}
        action = _stage3_compile_task_prose(draft, [a], proposed_template_action([a]))
        self.assertEqual(action['params']['tasks'][0]['context'], a['context'])

    def test_unknown_operator_cannot_enable_fallback(self):
        target = {'source_ids': [], 'outputs': {'typed_relation': ['answer']},
                  'operator_chain': [{'id': 'private', 'inputs': [], 'op': 'unknown_operation'}]}
        self.assertFalse(compile_task_contract(target, {})['fallback_supported'])

    def test_fallback_episode_round_checkpoints_resume_without_regeneration(self):
        from environment.tests.current_stage1 import current_scene
        from environment.tests.test_main_turn_dataset import reviewed_rows
        from environment.base_orchestration.trajectory_builder import build_episode
        class InvalidAuthor:
            def complete_json(self, system, payload): return {}
        scene = current_scene(0); rows = reviewed_rows(scene); saved = []
        first = build_episode(scene, rows, author=InvalidAuthor(), critic=Critic(),
            freeze_dispatch=True, allow_delegate_fallback=True,
            max_revisions=0, on_round=saved.append)
        self.assertEqual(first[3]['status'], 'accepted')
        self.assertTrue(any(x['turn']['generation_mode'] == 'contract_frozen_delegate_fallback' for x in saved))
        second = build_episode(scene, rows, author=InvalidAuthor(), critic=Critic(),
            freeze_dispatch=True, allow_delegate_fallback=True,
            max_revisions=0, resume_rounds=saved)
        self.assertEqual(first[:3], second[:3])

    def test_no_fallback_rejects_scene_and_publishes_no_partial_turns(self):
        from environment.tests.current_stage1 import current_scene
        from environment.tests.test_main_turn_dataset import reviewed_rows
        from environment.base_orchestration.trajectory_builder import build_episode

        class InvalidAuthor:
            def complete_json(self, system, payload):
                return {}

        scene = current_scene(0)
        saved = []
        episode, turns, audit, status = build_episode(
            scene, reviewed_rows(scene),
            author=InvalidAuthor(), critic=Critic(), freeze_dispatch=True,
            max_revisions=0, on_round=saved.append,
        )
        self.assertIsNone(episode)
        self.assertEqual(turns, [])
        self.assertEqual(audit, [])
        self.assertEqual(saved, [])
        self.assertEqual(status['status'], 'rejected')
        self.assertEqual(status['cause'], 'main_action_not_accepted')
        self.assertEqual(status['authoring']['mode'], 'llm_reviewed_contract_frozen')

    def test_completion_fallback_counts_as_failed_scene_in_strict_mode(self):
        from environment.tests.current_stage1 import current_scene
        from environment.tests.test_main_turn_dataset import reviewed_rows
        from environment.tests.test_stage3_natural_task_prose import (
            TaskProseAuthor, PassingCritic,
        )
        from environment.base_orchestration.trajectory_builder import build_episode

        class InvalidCompletionAuthor(TaskProseAuthor):
            def complete_json(self, system, payload):
                if 'pending_investigations' not in json.loads(payload):
                    return {}
                return super().complete_json(system, payload)

        scene = current_scene(0)
        saved = []
        episode, turns, audit, status = build_episode(
            scene, reviewed_rows(scene),
            author=InvalidCompletionAuthor(), critic=PassingCritic(),
            freeze_dispatch=True, max_revisions=0, on_round=saved.append,
        )
        self.assertIsNone(episode)
        self.assertEqual(turns, [])
        self.assertEqual(audit, [])
        self.assertEqual(status['status'], 'rejected')
        self.assertEqual(status['cause'], 'completion_not_llm_reviewed')
        self.assertEqual(
            status['authoring']['mode'],
            'teacher_grounded_completion_fallback',
        )
        self.assertTrue(saved)  # prior successful rounds remain checkpointed
        self.assertTrue(all(
            row['turn']['output']['action'] == 'delegate_task'
            for row in saved
        ))

    def test_public_overlap_uses_only_complete_returned_lists(self):
        from environment.base_orchestration.task_contract import public_identifier_join
        target = {'depends_on': ['left','right'], 'external_inputs': ['a','b'],
                  'operator_chain': [{'op': 'join', 'inputs': ['a','b'], 'left_on': 'case_id', 'right_on': 'case_id'}]}
        reports = {'left': {'finish_result': {'result': 'Dossier-A1, Dossier-A2, Dossier-X3'}},
                   'right': {'finish_result': {'result': 'Three dossiers match both clues: Dossier-B1, Dossier-B2 and Dossier-X3.;'}}}
        witness = public_identifier_join(target, reports)
        self.assertEqual(witness['shared_identifiers'], ['Dossier-X3'])
        reports['left']['finish_result']['result'] = 'For example Dossier-A1, but more records exist.'
        self.assertIsNone(public_identifier_join(target, reports))

    def test_review_backend_receives_required_issue_schema(self):
        from environment.base_orchestration.task_contract import complete_prose_review
        class SchemaBackend:
            def complete_json(self, system, payload, response_schema=None):
                self.schema = response_schema
                return {'verdict': 'pass'}
        backend = SchemaBackend()
        complete_prose_review(backend, 'review', '{}', structured=True)
        self.assertIn('issues', backend.schema['required'])
        self.assertIn('quote', backend.schema['properties']['issues']['items']['required'])

    def test_legacy_feedback_keeps_exact_claim_and_trusted_overlap(self):
        from environment.base_orchestration.task_contract import critic_repair_guidance
        a = assignment(); a['semantic_contract']['observed_identifier_join'] = {'shared_identifiers': ['Dossier-X3']}
        candidate = {'reasoning': 'We have six unique dossier identifiers.', 'params': {'tasks': []}}
        hints = critic_repair_guidance({'reason': "The claim 'We have six unique dossier identifiers' is incorrect."}, candidate, [a])
        self.assertTrue(any('six unique' in h for h in hints))
        self.assertTrue(any('Dossier-X3' in h for h in hints))

    def test_reasoning_cannot_invent_or_abbreviate_identifiers(self):
        from environment.base_orchestration.action_author import _stage3_task_prose_issues
        a = assignment()
        draft = {'reasoning': 'The project PRJ-ABC is ready for the next investigation.',
                 'tasks': [{'task_instruction': a['target_instruction'],
                            'context': 'Use the identified project to retrieve eligible events.'}]}
        issues = _stage3_task_prose_issues(draft, [a], prompt='The returned project is PRJ-ABC123.')
        self.assertIn('stage3_reasoning_invented_visible_identifier', issues)

    def test_handoff_shortening_never_creates_partial_identifiers(self):
        from environment.base_orchestration.action_author import _short, _STAGE3_IDENTIFIER
        text = 'Already returned identifier PRJ-ABCDEFGHIJKLMNO and supporting observations.'
        clipped = _short(text, 38)
        self.assertTrue(set(_STAGE3_IDENTIFIER.findall(clipped)) <= set(_STAGE3_IDENTIFIER.findall(text)))
        self.assertLessEqual(len(clipped), 38)

    def test_shortened_context_cannot_authorize_an_unseen_identifier(self):
        from environment.base_orchestration.action_author import _stage3_task_prose_issues
        a = assignment(); a['context'] = 'Previously returned result: PRJ-ABC...'
        draft = {'reasoning': 'Use project PRJ-ABC to request the eligible event records.',
                 'tasks': [{'task_instruction': a['target_instruction'],
                            'context': 'Use project PRJ-ABC to retrieve eligible events.'}]}
        issues = _stage3_task_prose_issues(draft, [a], prompt='The returned project is PRJ-ABC123.')
        self.assertIn('stage3_reasoning_invented_visible_identifier', issues)
        self.assertIn('task_0_invented_visible_identifier', issues)
