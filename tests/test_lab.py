import qtoolkit
import pytest

def test_metrics() -> None:
    rel = 1e-3

    zz_tt = 17003
    zz_tr = 731
    zz_rt = 346
    zz_rr = 16898

    xx_tt = 16584
    xx_tr = 538
    xx_rt = 190
    xx_rr = 17667

    zz = qtoolkit.BasisMetrics(
        c_00=zz_tt,
        c_01=zz_tr,
        c_10=zz_rt,
        c_11=zz_rr
    )

    assert zz.odd == pytest.approx(1077)
    assert zz.even == pytest.approx(33901)
    assert zz.total == pytest.approx(34978)
    assert zz.even_probability == pytest.approx(0.969209, rel=rel)
    assert zz.qber == pytest.approx(0.030791, rel=rel)
    assert zz.visibility == pytest.approx(0.938418, rel=rel)

    xx = qtoolkit.BasisMetrics(
        c_00=xx_tt,
        c_01=xx_tr,
        c_10=xx_rt,
        c_11=xx_rr
    )

    assert xx.odd == pytest.approx(728)
    assert xx.even == pytest.approx(34251)
    assert xx.total == pytest.approx(34979)
    assert xx.even_probability == pytest.approx(0.979188, rel=rel)
    assert xx.qber == pytest.approx(0.020812, rel=rel)
    assert xx.visibility == pytest.approx(0.958375, rel=rel)

    metrics = qtoolkit.BBM92Metrics(zz=zz, xx=xx)

    assert metrics.fidelity == pytest.approx(0.9484, rel=rel)
