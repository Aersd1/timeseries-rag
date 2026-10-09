import unittest
import torch
from v7.tests.test_variants import tiny_adapter
from v7.retrieval_finetune import RetrievalFineTune, PairedRetrievalAttention, quantile_loss, memory_quality


class FineTuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_identity_recursive_gradients_and_frozen_weights(self):
        torch.manual_seed(17)
        adapter = tiny_adapter()
        x = torch.randn(2,40)
        baseline = adapter.forecast(x,65)
        model = RetrievalFineTune(adapter,40,65)
        examples, weights, valid = torch.randn(2,5,105), torch.ones(2,5)/5, torch.ones(2,5,dtype=torch.bool)
        frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        for step in range(2):
            optimizer.zero_grad()
            with torch.autograd.graph.allow_mutation_on_saved_tensors():
                prediction = model(x,examples,weights,valid)
                if step==0:
                    torch.testing.assert_close(prediction,baseline,rtol=0,atol=0)
                loss = quantile_loss(prediction,torch.randn(2,65),x,torch.ones(2)*.1,model.module.quantile_levels,[24,65])
                loss.backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter,frozen[name],rtol=0,atol=0)
        model.eval()
        with torch.no_grad():
            a = model(x,examples,weights,valid)
            b = model(x,examples.flip(-1),weights,valid)
            self.assertFalse(torch.allclose(a,b))
            # Invalid memory is an exact no-memory fallback even after training.
            torch.testing.assert_close(model(x,examples,weights,valid&False),model(x),rtol=0,atol=0)
        saved = model.adaptation_state()
        second = RetrievalFineTune(tiny_adapter(),40,65)
        second.module.load_state_dict(model.module.state_dict())
        second.load_adaptation(saved)
        with torch.no_grad():
            torch.testing.assert_close(second(x,examples,weights,valid),a,rtol=0,atol=0)
        self.assertIsNone(model._memory)

    def test_plain_lora_has_no_memory_parameters(self):
        model = RetrievalFineTune(tiny_adapter(),40,16,use_retrieval=False)
        self.assertIsNone(model.retrieval)
        self.assertTrue(all('.a.' in name or '.b.' in name for name,p in model.named_parameters() if p.requires_grad))

    def test_gated_and_paired_recursive_training_reload_and_permutation(self):
        for fusion in ['gated','paired']:
            with self.subTest(fusion=fusion):
                torch.manual_seed(29)
                adapter = tiny_adapter()
                x, y = torch.randn(2,40), torch.randn(2,65)
                baseline = adapter.forecast(x,65)
                model = RetrievalFineTune(adapter,40,65,fusion=fusion)
                examples = torch.randn(2,5,105)
                weights = torch.softmax(torch.randn(2,5),-1)
                valid = torch.ones(2,5,dtype=torch.bool)
                frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
                for step in range(3):
                    optimizer.zero_grad()
                    with torch.autograd.graph.allow_mutation_on_saved_tensors():
                        prediction = model(x,examples,weights,valid)
                        if step==0:
                            torch.testing.assert_close(prediction,baseline,rtol=0,atol=0)
                        loss = quantile_loss(prediction,y,x,torch.ones(2)*.1,model.module.quantile_levels,[24,65])
                        loss.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    if step==2:
                        for prefix in ['retrieval.gate', 'retrieval.q', 'retrieval.output']:
                            self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
                                if n.startswith(prefix) and p.grad is not None),0)
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    a = model(x,examples,weights,valid)
                    gate = model.retrieval.last_gate
                    self.assertTrue(((gate>=0)&(gate<=1)).all())
                    permutation = torch.tensor([4,1,0,3,2])
                    torch.testing.assert_close(model(x,examples[:,permutation],weights[:,permutation],valid[:,permutation]),a)
                    torch.testing.assert_close(model(x,examples,weights,valid&False),model(x),rtol=0,atol=0)
                    changed = examples.clone(); changed[...,40:]+=2
                    self.assertFalse(torch.allclose(model(x,changed,weights,valid),a))
                    # Saturated zero gate gives the exact no-memory prediction.
                    model.retrieval.gate[-1].weight.zero_(); model.retrieval.gate[-1].bias.fill_(-1000)
                    torch.testing.assert_close(model(x,examples,weights,valid),model(x),rtol=0,atol=0)
                for n,p in model.named_parameters():
                    if n in frozen:
                        self.assertIsNone(p.grad); torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
                second = RetrievalFineTune(tiny_adapter(),40,65,fusion=fusion)
                second.module.load_state_dict(model.module.state_dict())
                second.load_adaptation(model.adaptation_state()); second.eval()
                with torch.no_grad():
                    torch.testing.assert_close(second(x,examples,weights,valid),model(x,examples,weights,valid),rtol=0,atol=0)

    def test_paired_key_depends_only_on_history_and_invalid_candidates_are_ignored(self):
        attention = PairedRetrievalAttention(64,40,65)
        examples = torch.randn(2,5,105); weights = torch.ones(2,5)/5
        valid = torch.ones(2,5,dtype=torch.bool); valid[:,4]=False
        changed = examples.clone(); changed[...,40:]+=10
        a, b = attention.memory(examples,weights,valid), attention.memory(changed,weights,valid)
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
        self.assertFalse(torch.equal(a[1],b[1]))
        model = RetrievalFineTune(tiny_adapter(),40,65,fusion='paired').eval()
        with torch.no_grad():
            model.retrieval.output.weight.normal_(std=.1)
            x = torch.randn(2,40); expected=model(x,examples,weights,valid)
            examples[:,4]+=10000
            torch.testing.assert_close(model(x,examples,weights,valid),expected,rtol=0,atol=0)
            self.assertTrue(torch.isfinite(memory_quality(x,examples,weights,valid&False,40)).all())

    def test_dropout_is_training_only_and_all_dropped_is_safe(self):
        for option in ['candidate_dropout','memory_dropout']:
            model = RetrievalFineTune(tiny_adapter(),40,65,fusion='paired',**{option:1.})
            x=torch.randn(2,40); examples=torch.randn(2,5,105)
            weights=torch.ones(2,5)/5; valid=torch.ones(2,5,dtype=torch.bool)
            with torch.no_grad():
                model.retrieval.output.weight.normal_(std=.1)
                model.train()
                torch.testing.assert_close(model(x,examples,weights,valid),model(x),rtol=0,atol=0)
                self.assertEqual(float(model.retrieval.last_gate.abs().sum()),0.)
                model.eval()
                a=model(x,examples,weights,valid); b=model(x,examples,weights,valid)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.assertFalse(torch.allclose(a,model(x)))


if __name__ == '__main__':
    unittest.main()
