import unittest
import numpy as np
from v7.history_length import observed_context,common_eligible,select_lengths,paired_interval,plateau_diagnostic


class HistoryLengthTests(unittest.TestCase):
    def test_context_stops_at_prediction_boundary_and_owns_storage(self):
        x=np.arange(7000,dtype=np.float32);past=observed_context(x,5000,4096)
        self.assertEqual(len(past),4096);self.assertEqual(past[-1],5095)
        x[5096:]=1e15
        np.testing.assert_array_equal(past,observed_context(x,5000,4096))
        x[:]=0
        self.assertEqual(past[-1],5095)

    def test_common_cohort_requires_whole_long_history_and_target(self):
        s=dict(runs=[(0,4500),(4600,10000)])
        self.assertFalse(common_eligible(s,5000))
        self.assertTrue(common_eligible(s,9000))
        self.assertFalse(common_eligible(dict(runs=[(0,10000)]),9900))

    def test_selection_uses_validation_and_ties_favor_shorter_history(self):
        scores={str(k):dict(mse=m,mae=0) for k,m in [(96,10),(512,8),(2016,6),(4096,7)]}
        blocks=[dict(metrics={str(k):dict(mse=m) for k,m in [(96,10),(512,9),(2016,11),(4096,10)]})]*4
        p=select_lengths(scores,blocks)
        self.assertEqual(p['primary_length'],2016)
        self.assertEqual(p['robust_secondary_length'],512)
        same={str(k):dict(mse=1) for k in (96,512,2016,4096)}
        self.assertEqual(select_lengths(same,[dict(metrics=same)])['primary_length'],96)

    def test_zero_gain_and_plateau_are_distinct_from_missing_evidence(self):
        data=dict(raw_y=np.zeros((6,96)),start=np.arange(6)*192)
        out=paired_interval(data,np.ones((6,96)),np.ones((6,96)),24)
        self.assertEqual(out['gain_percent'],0.)
        self.assertEqual(plateau_diagnostic(out)['classification'],'within_predeclared_one_percent_band')
        uncertain=dict(gain_percent=.1,exploratory_interval95=[-5.,6.])
        self.assertEqual(plateau_diagnostic(uncertain)['classification'],'insufficient_evidence_of_plateau')


if __name__=='__main__':unittest.main()
