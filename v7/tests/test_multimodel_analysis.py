import unittest
import numpy as np
from v7.analyze_multimodel_residual import holm,resampled_risks


class MultiModelAnalysisTests(unittest.TestCase):
    def test_holm_adjusts_complete_family_in_original_order(self):
        np.testing.assert_allclose(holm([.6,.001,.02]),[.6,.003,.04])

    def test_shared_seed_time_resampling_preserves_constant_paired_gain(self):
        counts=np.array([1,2,1,2]);risk=np.array([2.,1.])
        sums=np.repeat((counts[:,None]*risk)[None],3,axis=0)
        rng=np.random.default_rng(0);seeds=rng.integers(3,size=(50,3));weeks=rng.integers(4,size=(50,4))
        out=resampled_risks(sums,counts,seeds,weeks)
        np.testing.assert_allclose(out,np.repeat(risk[None],50,axis=0))


if __name__=='__main__':unittest.main()
