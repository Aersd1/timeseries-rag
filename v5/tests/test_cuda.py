"""Optional tiny CUDA/AMP correctness probe, without optimizer updates."""
import importlib.util
import unittest
from test_core import config


class CudaTests(unittest.TestCase):
    def test_amp_forward_backward_without_training(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('PyTorch unavailable')
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable; CPU-only CI/server')
        free,_ = torch.cuda.mem_get_info()
        if free < 512*1024*1024:
            self.skipTest('Less than 512 MiB free; avoid pressuring the device')
        from v5.model import UnifiedTokens,objective
        c=config(); model=UnifiedTokens(c).cuda()
        batch=dict(query=torch.randn(2,64,device='cuda'),
                   candidates=torch.randn(2,8,64,device='cuda'),
                   target=torch.randn(2,12,device='cuda'),
                   valid=torch.tensor([[True]*7+[False],[True]*8],device='cuda'),
                   past_distance=torch.rand(2,8,device='cuda'),
                   future_nmse=torch.rand(2,8,3,device='cuda'),
                   scale_floor=torch.full((2,),1e-6,device='cuda'))
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            loss,parts=objective(model,batch)
        self.assertTrue(torch.isfinite(loss))
        # Exercise scaling/backpropagation, but never step an optimizer.
        scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        scaler.scale(loss).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        torch.cuda.synchronize()
        del batch,model,loss,parts,scaler
        torch.cuda.empty_cache()


if __name__ == '__main__': unittest.main()
