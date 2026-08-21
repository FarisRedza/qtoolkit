import qtoolkit
import numpy as np
import pytest


def test_timetag_data_from_file(tmp_path) -> None:
    file_path = tmp_path.joinpath('timetags.txt')

    file_path.write_text(
        '941575226770542 6\n'
        '941575227172420 0\n'
        '941575227234390 6\n'
        '941575227678840 6\n'
        '941575228068663 3\n'
        '941575228250870 4\n'
        '941575230136717 6\n'
        '941575230665605 2\n'
        '941575230717349 7\n'
        '941575230717507 1'
    )

    timetag_data = qtoolkit.TimetagData.from_file(
        file_path=file_path
    )

    np.testing.assert_array_equal(
        timetag_data.timetags,
        np.array([
            941575226770542,
            941575227172420,
            941575227234390,
            941575227678840,
            941575228068663,
            941575228250870,
            941575230136717,
            941575230665605,
            941575230717349,
            941575230717507
            ],
            dtype=np.int64
        ),
    )

    np.testing.assert_array_equal(
        timetag_data.channels,
        np.array(
            [6, 0, 6, 6, 3, 4, 6, 2, 7, 1],
            dtype=np.int8
        ),
    )

    assert timetag_data.file_path == file_path

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
