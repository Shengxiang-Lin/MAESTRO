"""Regressions for semantic drift observed in real Qwen/O1 smoke trajectories."""
import copy
import json
import unittest
from unittest.mock import Mock
from environment.tests.test_gold_trajectory import fixture
from environment.trajectory_simulation.compiler import compile_gold_solution,private_guidance
from environment.trajectory_simulation.pipeline import synthesize_one,GoldTrajectoryError
from environment.trajectory_simulation.backend import PassthroughPlanner,PassthroughExecutor,LLMExecutor
from environment.trajectory_simulation.realization import feedback_variants,realize_feedback
from environment.trajectory_simulation.tools import tool_profile,ToolProfile

def branch_fixture():
    q,r,t=fixture()
    r['program']['nodes']=[
        {'node_id':0,'op':'FIND','dependencies':[],'inputs':['entity:A']},
        {'node_id':1,'op':'RELATE','dependencies':[0],'inputs':['r:first','forward']},
        {'node_id':2,'op':'FILTER_NUMBER','dependencies':[1],'inputs':['r:score','<CMP:=>','1']},
        {'node_id':3,'op':'FIND_ALL','dependencies':[],'inputs':[]},
        {'node_id':4,'op':'FILTER_NUMBER','dependencies':[3],'inputs':['r:score','<CMP:>>','0']},
        {'node_id':5,'op':'INTERSECT','dependencies':[2,4],'inputs':[]},
        {'node_id':6,'op':'COUNT','dependencies':[5],'inputs':[]}]
    r['world']['facts'][1].update(subject='B',predicate='r:score',object=1,object_kind='number')
    t.answer=1;t.sources[1].content='Bob score 1'
    return q,r,t

class BoundRealizationTests(unittest.TestCase):
    def test_scene_maps_audio_to_native_o1_name(self):
        profile=tool_profile()
        self.assertEqual(profile.for_resource({'source_type':'audio_transcript','visibility':'attached'}),('ParseAudioAction',))
        self.assertNotIn('AudioAnalysisAction',profile.available)
        self.assertEqual(profile.for_resource({'source_type':'image_report','visibility':'discoverable'}),('GoogleSearchAction','ImageAnalysisAction'))
        with self.assertRaisesRegex(ValueError,'unsupported source modality'):
            profile.for_resource({'source_type':'unregistered'})
        # Mapping belongs to the scene, not an if/else inside the compiler.
        custom=ToolProfile('custom',{'audio':('ListenAction',)})
        self.assertEqual(custom.for_resource({'source_type':'audio_transcript'}),('ListenAction',))

    def test_independent_terminal_branch_is_scheduled_before_the_join(self):
        gold=compile_gold_solution(*branch_fixture())
        self.assertIn('fragment:branch:4',gold['waves'][0])
        self.assertIn('fragment:0',gold['waves'][0])
        terminal=gold['fragments'][-1]
        self.assertEqual([o['op'] for o in terminal['operations']],['INTERSECT','COUNT'])
        self.assertEqual(set(terminal['depends_on']),{'fragment:1','fragment:branch:4'})

    def test_completed_inputs_are_explicit_and_empty_inputs_stay_empty(self):
        episode=synthesize_one(*branch_fixture(),PassthroughPlanner(),PassthroughExecutor())
        second=episode['trajectory'][1]['params']['tasks'][0]
        self.assertIn('Attempt 1, subtask',second['task_instruction'])
        self.assertNotIn('all supplied records',second['task_instruction'])
        q,r,t=fixture();r['world']['facts'][0]['object']='A';t.answer=[]
        # Compiler still executes true empty intermediates; reports cannot rediscover entities.
        episode=synthesize_one(q,r,t,PassthroughPlanner(),PassthroughExecutor())
        self.assertTrue(episode['validation']['accepted'])

    def test_all_records_scope_cannot_replace_completed_input(self):
        class Drift(PassthroughPlanner):
            def decide(self,payload,prompt):
                action=super().decide(payload,prompt)
                if action['action']=='delegate_task' and payload['completed_subtask_contracts']:
                    task=action['params']['tasks'][0]
                    task['task_instruction']='Start from all supplied records. Keep entities with score = 1.'
                return action
        with self.assertRaisesRegex(GoldTrajectoryError,'outside_bound_execution_contract'):
            synthesize_one(*branch_fixture(),Drift(),PassthroughExecutor(),max_attempts=1)

    def test_latent_find_name_is_not_given_to_main_before_discovery(self):
        q,r,t=fixture();q['question']='Find the entity identified by the question clues.'
        gold=compile_gold_solution(q,r,t)
        guidance=private_guidance(gold,set())
        self.assertNotIn('Alice',json.dumps(guidance))
        # A hidden FIND with no public identifying clue is now quarantined,
        # rather than filled from the private target identity.
        with self.assertRaises(GoldTrajectoryError) as caught:
            synthesize_one(q,r,t,PassthroughPlanner(),PassthroughExecutor())
        self.assertEqual(caught.exception.stage, 'public_executability')
        self.assertEqual(caught.exception.issues, ['latent_find_missing_public_clue'])
        self.assertEqual(
            caught.exception.audit['public_preflight']['detail'],
            'missing concrete identification clue',
        )

    def test_wrong_explanation_is_rejected_even_when_symbolic_answer_is_fixed(self):
        class Drift(PassthroughExecutor):
            def respond(self,payload,prompt):
                result=super().respond(payload,prompt)
                result['trace_summary']+=' Excluded records with 1511 because they did not satisfy != 440.'
                return result
        with self.assertRaisesRegex(GoldTrajectoryError,'outside_bound_execution_contract'):
            synthesize_one(*fixture(),PassthroughPlanner(),Drift(),max_attempts=1)

    def test_reports_match_real_trace_style_and_do_not_expose_teacher(self):
        episode=synthesize_one(*fixture(dependent=True),PassthroughPlanner(),PassthroughExecutor())
        trace=episode['trajectory'][0]['result']['subtask_results'][0]['trace_summary']
        self.assertIn('* Inspected',trace);self.assertIn('* Followed',trace)
        self.assertNotIn('authoritative',json.dumps(episode['trajectory']).lower())
        self.assertIn('Reused the earlier observations',episode['trajectory'][-2]['result']['subtask_results'][0]['trace_summary'])
        for term in ['private fragment graph','previous attempt failed','SymbolicExecutor']:
            self.assertNotIn(term,json.dumps(episode['trajectory']))

    def test_public_reasoning_cannot_contain_retry_or_teacher_details(self):
        class Leak(PassthroughPlanner):
            def decide(self,payload,prompt):
                result=super().decide(payload,prompt)
                result['reasoning']='The private fragment graph says the previous attempt failed.'
                return result
        with self.assertRaisesRegex(GoldTrajectoryError,'invalid_public_reasoning'):
            synthesize_one(*fixture(),Leak(),PassthroughExecutor(),max_attempts=1)

    def test_scoped_literal_surfaces_are_used_in_conditions(self):
        q,r,t=branch_fixture()
        t.hidden_metadata['literal_surfaces_by_predicate']={'r:score':{'1':'one unit'}}
        g=compile_gold_solution(q,r,t)
        op=next(o for f in g['fragments'] for o in f['operations'] if o['node_id']==2)
        self.assertEqual(op['inputs'][-1],'one unit')

    def test_rendered_oracle_cannot_disagree_with_program(self):
        q,r,t=fixture();t.answer=['Invented answer']
        with self.assertRaisesRegex(ValueError,'rendered final answer disagrees'):
            compile_gold_solution(q,r,t)

    def test_llm_cannot_inject_free_text_in_style_controls(self):
        gold=compile_gold_solution(*fixture());fragment=gold['fragments'][0]
        payload={'fragment':fragment}
        with self.assertRaisesRegex(ValueError,'invalid report language plan'):
            realize_feedback(payload,{'report_style':'stepwise','summary':'Invented result'})
        self.assertEqual(realize_feedback(payload,{'report_style':'stepwise'}),feedback_variants(fragment)[1])

if __name__=='__main__':unittest.main()
