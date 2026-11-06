"""Behavioral regressions for the reorganized construction/authoring pipeline."""
import random
import unittest
from environment.task_construction.ir import ProgramIR,ProgramNode as N
from environment.task_construction.binding.program_binder import BindingPlan
from environment.task_construction.binding.predicate_binder import AttributeBinding,RelationBinding
from environment.task_construction.deep_dependency.composer import _terminal,compose_deep_program
from environment.task_construction.synthesis.world_sampler import WorldSampler,unresolved_slot
from environment.task_construction.executor import SymbolicExecutor
from environment.task_construction.quality.diversity import BatchDiversityTracker
from environment.question_generation.validators import preflight


class ConstructionRegressions(unittest.TestCase):
    def test_terminal_keeps_scalar_dependency(self):
        template=ProgramIR([N(0,'FIND'),N(1,'QUERY_ATTR',(0,),('<ATTR>',)),
            N(2,'VERIFY_YEAR',(1,),('<CMP:=>','<YEAR>'))])
        result=_terminal([N(0,'FIND')],template,random.Random(1))
        self.assertEqual([n.op for n in result],['FIND','QUERY_ATTR','VERIFY_YEAR'])

    def test_string_slot_becomes_concrete_in_facts_and_program(self):
        p=ProgramIR([N(0,'FIND',(),('type:schema:Person',)),
            N(1,'FILTER_STRING',(0,),('schema:address','<CMP:=>','<SYMBOL>')),
            N(2,'QUERY_NAME',(1,))])
        binding=AttributeBinding('schema:address','schema:Person','string')
        sampler=WorldSampler(seed=17)
        world=sampler.synthesize(BindingPlan(p,{0:'schema:Person',1:'schema:Person'},attribute_bindings={1:binding}))
        concrete=sampler.concrete_program
        self.assertFalse(any(unresolved_slot(v) for n in concrete.nodes for v in n.inputs))
        self.assertTrue(SymbolicExecutor().execute(concrete,world))
        self.assertEqual(str(world.facts[0].object),concrete.nodes[1].inputs[2])

    def test_strict_numeric_filter_is_satisfied(self):
        p=ProgramIR([N(0,'FIND',(),('type:schema:Person',)),
            N(1,'FILTER_NUMBER',(0,),('schema:height','<CMP:>>','<NUMBER>')),N(2,'COUNT',(1,))])
        b=AttributeBinding('schema:height','schema:Person','number')
        sampler=WorldSampler(seed=17)
        world=sampler.synthesize(BindingPlan(p,{0:'schema:Person',1:'schema:Person'},attribute_bindings={1:b}))
        self.assertGreater(SymbolicExecutor().execute(sampler.concrete_program,world),0)

    def test_intersection_constructs_both_branches(self):
        p=ProgramIR([N(0,'FIND',(),('type:schema:Person',)),
            N(1,'RELATE',(0,),('r:left','forward')),N(2,'RELATE',(0,),('r:right','forward')),
            N(3,'INTERSECT',(1,2)),N(4,'QUERY_NAME',(3,))])
        types={i:'schema:Person' for i in range(4)}
        rels={1:RelationBinding('r:left','schema:Person','schema:Person','forward'),
              2:RelationBinding('r:right','schema:Person','schema:Person','forward')}
        sampler=WorldSampler(seed=17)
        world=sampler.synthesize(BindingPlan(p,types,rels))
        trace=SymbolicExecutor().execute_trace(sampler.concrete_program,world)
        self.assertTrue(trace[3])
        self.assertEqual(trace[1]&trace[2],trace[3])
        self.assertEqual({f.predicate for f in world.facts},{'r:left','r:right'})

    def test_verify_materializes_scalar_target_without_mutating_plan(self):
        p=ProgramIR([N(0,'FIND',(),('type:schema:Person',)),
            N(1,'QUERY_ATTR',(0,),('schema:birthDate',)),
            N(2,'VERIFY_YEAR',(1,),('<CMP:=>','<YEAR>'))])
        b=AttributeBinding('schema:birthDate','schema:Person','year',value_min=1970,value_max=2020)
        for seed in (17,18):
            sampler=WorldSampler(seed=seed)
            world=sampler.synthesize(BindingPlan(p,{0:'schema:Person'},attribute_bindings={1:b}))
            self.assertIsInstance(SymbolicExecutor().execute(sampler.concrete_program,world),bool)
            self.assertFalse(unresolved_slot(sampler.concrete_program.nodes[2].inputs[1]))
        self.assertEqual(p.nodes[2].inputs[1],'<YEAR>')

    def test_operator_type_validation_rejects_entity_verify(self):
        from environment.task_construction.binding.graph_grounding import validate_program_structure
        from environment.task_construction.binding.program_binder import BindingError
        p=ProgramIR([N(0,'FIND'),N(1,'VERIFY_YEAR',(0,),('<CMP:=>','<YEAR>'))])
        with self.assertRaises(BindingError):validate_program_structure(p)

    def test_caps_depend_on_requested_batch(self):
        tracker=BatchDiversityTracker(target=100)
        before=tracker.caps()
        tracker.add(topology='t',predicates=['p'],type_pairs=['a'],terminal_family='count')
        self.assertEqual(before,tracker.caps())
        self.assertGreater(before.predicate,3)

    def test_placeholder_is_rejected(self):
        q='Which entities have address equal to <SYMBOL>?'
        result=preflight({'draft_question':q},q)
        self.assertFalse(result.accepted)
        self.assertIn('synthetic_identifier_leak',result.issues)



if __name__ == "__main__":
    unittest.main()
