"""Fresh public replay and fail-closed training export regressions."""
import copy,json,unittest
from unittest.mock import Mock,patch
from environment.tests.test_gold_trajectory import fixture
from environment.tests.test_bound_realization import branch_fixture
from environment.trajectory_simulation.pipeline import synthesize_one,protect_paths,_stage
from environment.trajectory_simulation.backend import PassthroughPlanner,PassthroughExecutor,LLMPlanner
from environment.trajectory_simulation.evidence import read_packet,verify_known_packet,MARKER
from environment.trajectory_simulation.replay import replay_task
from environment.trajectory_simulation.exporter import export_episode,validate_public_episode
from environment.trajectory_simulation.eligibility import check_question,digest
from environment.trajectory_simulation.compiler import compile_gold_solution,private_guidance

def cached_height_fixture():
    q,r,t=fixture(terminal_evidence=True)
    r['world']['facts'][-1]['source_id']='s1'
    r['world']['sources']['s1']['fact_ids'].append('f3');del r['world']['sources']['s3']
    q['capability_world']['resources']=q['capability_world']['resources'][:2]
    t.sources=t.sources[:2];t.sources[0].content+='; Bob height 175'
    return q,r,t

def episode():return synthesize_one(*cached_height_fixture(),PassthroughPlanner(),PassthroughExecutor())

def mark_reviewed(row):
    row=copy.deepcopy(row);row['input_eligibility']['reviewed']=True;row['validation']['training_eligible']=True
    return row

class PublicQualityTests(unittest.TestCase):
    def test_reused_measurement_is_available_to_fresh_subagent(self):
        row=episode();step=row['trajectory'][-2];task=step['params']['tasks'][0]
        cache=step['state_before']['evidence_store']
        packet=read_packet(task);expanded=verify_known_packet(task,cache)
        self.assertEqual(task['tools'],['ExecuteCodeAction'])
        self.assertEqual(packet['version'],2)
        self.assertTrue(any(f['predicate']=='height' and f['object']==175
                            for record in expanded['known_records'] for f in record['claims']))
        self.assertLess(len(task['context']),1700)
        self.assertIn('First reference',task['context'])
        self.assertEqual(replay_task(task,[],public_cache=cache)['result'],[175])
        self.assertEqual(row['trajectory'][0]['state_before']['evidence_store'],{})
        self.assertTrue(step['state_before']['evidence_store']);validate_public_episode(row,require_training=False)

    def test_missing_cached_measurement_cannot_be_filled_from_gold(self):
        row=mark_reviewed(episode());task=row['trajectory'][-2]['params']['tasks'][0];packet=read_packet(task)
        packet['known_resource_ids'].append('future_unpublished_source')
        task['context']=task['context'].split(MARKER)[0]+MARKER+json.dumps(packet)
        with self.assertRaisesRegex(ValueError,'unpublished or altered cached evidence'):export_episode(row)

    def test_parallel_siblings_cannot_see_future_results(self):
        row=mark_reviewed(episode());first=row['trajectory'][0]
        self.assertEqual(len(first['params']['tasks']),2);self.assertEqual(first['state_before']['evidence_store'],{})
        task=first['params']['tasks'][1];packet=read_packet(task)
        packet['completed_inputs']['Attempt 1, subtask 1']={'value':['Bob'],'answer_kind':'entity_set'}
        task['context']=task['context'].split(MARKER)[0]+MARKER+json.dumps(packet)
        with self.assertRaisesRegex(ValueError,'future or altered completed input'):export_episode(row)

    def test_export_uses_legacy_schema_and_excludes_private_supervision(self):
        row=mark_reviewed(episode());row['gold_supervision']['canary']='PRIVATE_CANARY_123';records=export_episode(row)
        self.assertEqual(len(records),len(row['trajectory']));self.assertNotIn('PRIVATE_CANARY_123',json.dumps(records))
        self.assertNotIn('gold_supervision',records[0]['input']);self.assertEqual(records[-1]['macro_label'],'complete')
        self.assertEqual(records[0]['macro_label'],'advance');self.assertIn('PUBLIC EVIDENCE STORE',records[-1]['input'])
        self.assertEqual(json.loads(records[-1]['target_json'])['action'],'complete')

    def test_offline_reference_is_not_clean_training_data(self):
        with self.assertRaisesRegex(ValueError,'eligibility review'):export_episode(episode())

    def test_export_revalidates_late_result_and_state_fields(self):
        row=mark_reviewed(episode());row['trajectory'][-1]['params']['answer']='99999'
        with self.assertRaisesRegex(ValueError,'final answer contradicts'):export_episode(row)
        row=mark_reviewed(episode());row['trajectory'][0]['state_before']['private_gold_guidance']={'answer':'secret'}
        with self.assertRaisesRegex(ValueError,'unexpected public state'):export_episode(row)

    def test_export_rejects_changed_question(self):
        row=mark_reviewed(episode());row['question']+=' an extra condition'
        with self.assertRaisesRegex(ValueError,'question changed'):export_episode(row)

    def test_frozen_inputs_cannot_be_output_paths(self):
        with self.assertRaisesRegex(ValueError,'frozen input'):protect_paths(['input.jsonl'],['input.jsonl'])
        with self.assertRaisesRegex(ValueError,'distinct'):protect_paths(['input.jsonl'],['out.jsonl','out.jsonl'])

    def test_audited_review_is_bound_to_exact_inputs(self):
        q,r,t=fixture();g=compile_gold_solution(q,r,t)
        rule={'task_id':q['task_id'],'question_sha256':digest(q['question']),'program_sha256':digest(r['program']),'reason':'audited mismatch'}
        with patch('pathlib.Path.is_file',return_value=True), patch('pathlib.Path.read_text',return_value=json.dumps([rule])):
            self.assertFalse(check_question(g,r,PassthroughPlanner())['eligible'])
            g['question']+=' revised';result=check_question(g,r,PassthroughPlanner())
            self.assertTrue(result['eligible']);self.assertFalse(result['reviewed'])

    def test_missing_manual_review_file_does_not_fake_training_eligibility(self):
        q,r,t=fixture();g=compile_gold_solution(q,r,t)
        with patch('pathlib.Path.is_file',return_value=False):
            status=check_question(g,r,PassthroughPlanner())
        self.assertTrue(status['eligible'])
        self.assertFalse(status['reviewed'])
        self.assertEqual(status['method'],'offline_structure_only')

    def test_uncertain_question_is_quarantined(self):
        q,r,t=fixture();g=compile_gold_solution(q,r,t);role=Mock(offline=False,backend=Mock())
        role.check_question.return_value={'decision':'uncertain','issues':[]}
        result=check_question(g,r,role);self.assertFalse(result['eligible']);self.assertTrue(result['reviewed'])

    def test_public_gold_preflight_rejects_unidentified_root(self):
        from environment.trajectory_simulation.public_preflight import audit_public_executability
        from environment.trajectory_simulation.pipeline import GoldTrajectoryError
        q,r,t=fixture()
        q['question']='Which entity satisfies the composed relationship checks?'
        gold=compile_gold_solution(q,r,t,schedule='compact')
        audit=audit_public_executability(gold)
        self.assertFalse(audit['publicly_executable'])
        self.assertEqual(audit['reason'],'latent_find_missing_public_clue')
        self.assertIn('fragment_id',audit)
        self.assertIn('output_node_id',audit)
        with self.assertRaises(GoldTrajectoryError) as error:
            synthesize_one(q,r,t,PassthroughPlanner(),PassthroughExecutor(),schedule='compact')
        self.assertEqual(error.exception.stage,'public_executability')
        self.assertEqual(error.exception.audit['public_preflight']['reason'],
                         'latent_find_missing_public_clue')

    def test_public_gold_preflight_accepts_fully_replayable_plan(self):
        from environment.trajectory_simulation.public_preflight import audit_public_executability
        q,r,t=fixture()
        gold=compile_gold_solution(q,r,t,schedule='compact')
        report=audit_public_executability(gold)
        self.assertTrue(report['publicly_executable'])
        self.assertEqual(report['checked_fragments'],len(gold['fragments']))

    def test_plan_reports_compilation_separately_from_public_readiness(self):
        import tempfile
        from pathlib import Path
        from environment.trajectory_simulation.pipeline import plan_trajectories_jsonl
        q,r,t=fixture()
        q['question']='Which entity satisfies the composed relationship checks?'
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'plan.jsonl'
            inputs=Path(temp)/'realized.jsonl'
            inputs.write_text(json.dumps(q)+'\n',encoding='utf-8')
            with patch('environment.trajectory_simulation.pipeline._world_map',
                       return_value={q['task_id']:(r,t)}):
                summary=plan_trajectories_jsonl(
                    Path(temp)/'world.jsonl',inputs,target,schedule='compact')
            self.assertEqual(summary['compiled'],1)
            self.assertEqual(summary['ready'],0)
            self.assertEqual(summary['quarantined'],1)
            item=json.loads(target.read_text().strip())
            self.assertEqual(item['status'],'quarantined')
            self.assertEqual(item['public_executability']['reason'],
                             'latent_find_missing_public_clue')

    def test_compact_fusion_preserves_parallel_branches(self):
        row=synthesize_one(*branch_fixture(),PassthroughPlanner(),PassthroughExecutor(),schedule='compact')
        self.assertEqual(row['trajectory'][-1]['params']['answer'],'1')
        self.assertEqual(len(row['trajectory'][0]['params']['tasks']),2);self.assertEqual(len(row['trajectory']),3)
        validate_public_episode(row,require_training=False)

    def test_successful_attempts_retain_retry_count(self):
        audit=[];count=[]
        def call(issues):count.append(1);return {'okay':len(count)>1}
        _stage(call,lambda v:(v['okay'],[] if v['okay'] else ['retry']),stage='probe',max_attempts=3,attempt_audit=audit)
        self.assertEqual(audit[0]['attempts_used'],2)

    def test_template_styles_are_reproducible_without_generation_requests(self):
        q,r,t=fixture();g=compile_gold_solution(q,r,t);backend=Mock();role=LLMPlanner(backend);role.style_provider='template'
        payload={'public_state':{'question':q['question']},'gold_guidance':private_guidance(g,set()),'completed_subtask_contracts':[]}
        self.assertEqual(role.decide(payload,''),role.decide(payload,''));backend.complete_json.assert_not_called()

if __name__=='__main__':unittest.main()
