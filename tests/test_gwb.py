import unittest
import numpy as np
from agu_james.analysis.gnn.tc_gwb import integrate_gwb, G

class GWBTests(unittest.TestCase):
    def test_solid_rotation_and_old_counterexample(self):
        omega, f = 2e-4, 1e-4
        r = np.arange(12.5, 526, 25) * 1000
        for R in (100000., 500000., 93750., 12500., 5000.):
            value, reason = integrate_gwb(r, omega*r, f, R)
            self.assertEqual(reason, 'ok')
            self.assertAlmostEqual(value, (omega**2 + f*omega)*R**2/(2*G), places=10)
        q = (omega**2 + f*omega)*r
        old = np.trapezoid(q[:5], r[:5])/G
        correct = integrate_gwb(r, omega*r, f, 100000.)[0]
        self.assertAlmostEqual(old/correct, 1.25)

    def test_zero_and_missing(self):
        r = np.arange(12.5, 526, 25)*1000
        self.assertEqual(integrate_gwb(r, r*0, 1e-4, 100000), (0., 'ok'))
        for index, reason in [(0, 'missing_first_ring'), (2, 'missing_required_ring'), (4, 'missing_required_ring')]:
            v = r*1e-4; v[index] = np.nan
            value, status = integrate_gwb(r, v, 1e-4, 100000)
            self.assertTrue(np.isnan(value)); self.assertEqual(status, reason)
        self.assertEqual(integrate_gwb(r[:4], r[:4], 0., 100000)[1], 'missing_endpoint_support')
        v = r*1e-4; v[5] = np.nan
        self.assertEqual(integrate_gwb(r,v,1e-4,100000)[1], 'ok')

    def test_nonlinear_convergence(self):
        # V=a*r^2: q=a^2*r^3 + f*a*r^2 has an analytic integral.
        a, f, R = 1e-9, 1e-4, 100000.
        truth = (a*a*R**4/4 + f*a*R**3/3)/G
        errors = []
        for width in (25000.,12500.,6250.,3125.):
            r = np.arange(width/2,R+width,width)
            errors.append(abs(integrate_gwb(r,a*r*r,f,R)[0]-truth))
        self.assertTrue(all(b < a/3 for a,b in zip(errors,errors[1:])))

    def test_original_modules_agree(self):
        from agu_james.analysis.gnn import tc_gwb
        from agu_james.analysis.pangu import axisymmetric_gwb_metrics_all as pangu
        lat=np.linspace(18,22,81);lon=np.linspace(118,122,81)
        yy,xx=np.meshgrid(lat,lon,indexing='ij');u=-(yy-20)*5;v=(xx-120)*5;z=np.ones_like(u)*30000
        _, gnn=tc_gwb.compute_delta_z(u,v,z,lat,lon,20,120,25,[100,500])
        other=pangu.compute_gwb_multi(z/G,u,v,lat,lon,20,120,pangu.Config())
        np.testing.assert_allclose(gnn,[other['delta_z_gwb_100'],other['delta_z_gwb_500']],equal_nan=True)

if __name__ == '__main__': unittest.main()
