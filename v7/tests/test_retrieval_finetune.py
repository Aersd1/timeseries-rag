import unittest
import torch
from v7.tests.test_variants import tiny_adapter
from v7.retrieval_finetune import RetrievalFineTune, quantile_loss


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


if __name__ == '__main__':
    unittest.main()
