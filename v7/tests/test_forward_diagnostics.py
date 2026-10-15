import unittest
import numpy as np
from v7.forward_diagnostics import compare,adoption_diagnostic


class ForwardDiagnosticTests(unittest.TestCase):
    def test_exact_fallback_is_not_called_positive_stability(self):
        y=np.ones((40,96));base=np.zeros_like(y);starts=np.repeat(np.arange(20)*192,2)
        c=compare(base,base,y,starts,24.,100)
        self.assertEqual(c['gain_percent'],0.);np.testing.assert_array_equal(c['interval95'],[0.,0.]);self.assertFalse(c['meets_prespecified_stability'])

    def test_all_four_windows_must_improve_and_synchronous_variables_cluster(self):
        y=np.ones((40,96));base=np.zeros_like(y);forecast=np.ones_like(y)*.5;starts=np.repeat(np.arange(20)*192,2)
        c=compare(base,forecast,y,starts,24.,100);self.assertTrue(c['meets_prespecified_stability']);self.assertLess(c['weekly_clusters'],len(starts))
        forecast[-10:]=-1.;c=compare(base,forecast,y,starts,24.,100);self.assertFalse(c['meets_prespecified_stability'])

    def test_no_adoption_preserves_random_control_and_zero_contributions(self):
        y=np.ones((4,96));base=np.zeros_like(y);candidate=np.ones_like(y);mask=np.zeros((4,12),bool)
        c=adoption_diagnostic(base,candidate,base,mask,y,np.array([0,0,1,1]),1.,10)
        self.assertEqual(c['matched_coverage_random_mse'],1.);self.assertEqual(c['gain_contribution'],0.);self.assertEqual(c['harm_contribution'],0.)

    def test_hourly_week_is_exactly168_observations(self):
        base=np.zeros((4,96));y=np.ones_like(base)
        c=compare(base,base,y,np.arange(4)*168,24.,100)
        self.assertEqual(c['weekly_clusters'],4)

    def test_perfect_baseline_and_identical_fallback_have_zero_gain(self):
        y=np.zeros((4,96));c=compare(y,y,y,np.arange(4)*168,24.,100)
        self.assertEqual(c['gain_percent'],0.);np.testing.assert_array_equal(c['interval95'],[0.,0.]);self.assertFalse(c['meets_prespecified_stability'])


if __name__=='__main__':unittest.main()
