import unittest
import numpy as np
from v7.rag_effectiveness import transfer,old_case_check,improvement_identity
from v7.sensitivity_rag_effectiveness import remove_shared_direction,demean


def inputs():
    rng=np.random.default_rng(8);n=4
    x=rng.normal(size=(n,96));past=rng.normal(size=(n,10,96))*np.linspace(.5,2,10)[None,:,None]
    future=rng.normal(size=past.shape)+np.arange(10)[None,:,None]/10
    base=np.zeros((n,96));q=np.repeat(base[:,None],9,axis=1)
    return dict(x=x,quantiles=q,probe=np.zeros((n,9,32)),sid=np.arange(n),start=np.full(n,9000),
        past=past,future=future,prior=np.zeros_like(past),starts=np.tile(np.arange(10)*192,(n,1)),
        train_std=np.ones(n),base=base,length=2016)


class EffectivenessTests(unittest.TestCase):
    def test_history_really_changes_forecast_with_identical_futures(self):
        d=inputs();before=transfer(**d)
        # Swap which HISTORY is paired to fixed residual values, within each5.
        d['past']=d['past'].copy();d['past'][:,:5]=np.roll(d['past'][:,:5],1,axis=1)
        after=transfer(**d)
        self.assertGreater(np.max(abs(before['residual']-after['residual'])),1e-4)

    def test_joint_case_order_is_irrelevant_but_pair_derangement_is_not(self):
        d=inputs();before=transfer(**d)
        perm=np.r_[np.arange(4,-1,-1),np.arange(9,4,-1)]
        reordered=dict(d)
        for k in ('past','future','prior','starts'):reordered[k]=d[k][:,perm]
        np.testing.assert_allclose(transfer(**reordered)['residual'],before['residual'],atol=1e-12)
        self.assertGreater(np.max(abs(transfer(**d,arm='deranged_1')['residual']-before['residual'])),1e-4)

    def test_query_target_not_required_and_incomplete_cases_rejected(self):
        d=inputs();result=transfer(**d)
        self.assertTrue(np.isfinite(result['features']).all())
        d['starts']=np.full_like(d['starts'],8999)
        with self.assertRaises(ValueError):transfer(**d)

    def test_old_case_check_excludes_whole_overlapping_episodes(self):
        d=inputs();starts=np.tile(np.arange(10)*10,(4,1))
        gain,valid=old_case_check(d['past'],d['future'],d['prior'],starts,d['train_std'])
        np.testing.assert_array_equal(valid,0);np.testing.assert_array_equal(gain,0)

    def test_exact_improvement_accounting(self):
        y=np.ones((3,96));b=np.zeros_like(y);p=np.ones_like(y)*.25
        alignment,energy,gain=improvement_identity(y,b,p)
        np.testing.assert_allclose(gain,.4375)
        np.testing.assert_allclose(alignment-energy,gain)

    def test_direction_control_matches_energy_and_uses_no_target(self):
        rng=np.random.default_rng(2);r=rng.normal(size=(4,96));s=r+rng.normal(size=(4,96))
        orthogonal,matched=remove_shared_direction(r,s)
        np.testing.assert_allclose((orthogonal*s).sum(1),0,atol=1e-12)
        np.testing.assert_allclose((orthogonal**2).sum(1),(matched**2).sum(1),atol=1e-12)
        a,b=remove_shared_direction(r,np.zeros_like(r))
        np.testing.assert_array_equal(a,r);np.testing.assert_allclose(b,r)

    def test_two_way_controls_remove_variable_and_time_effects(self):
        sid=np.repeat(np.arange(3),4);start=np.tile(np.arange(4),3)
        confounded=10*sid+2*start
        np.testing.assert_allclose(demean(confounded,sid,start),0,atol=1e-10)


if __name__=='__main__':unittest.main()
