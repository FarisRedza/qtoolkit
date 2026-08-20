def qber(
        correct: float,
        incorrect: float
) -> float:
    return incorrect / (incorrect + correct)

def qber_from_coincidences(
        c_00: float,
        c_01: float,
        c_10: float,
        c_11: float,
        correlated: bool = True
) -> float:
    if correlated:
        correct = c_00 + c_11
        incorrect = c_01 + c_10
    else:
        correct = c_01 + c_10
        incorrect = c_00 + c_11

    return qber(correct, incorrect)

def qz(
        c_hh: float,
        c_hv: float,
        c_vh: float,
        c_vv: float,
        correlated: bool = True
) -> float:
    return qber_from_coincidences(
        c_00=c_hh,
        c_01=c_hv,
        c_10=c_vh,
        c_11=c_vv,
        correlated=correlated
    )

def qx(
        c_dd: float,
        c_da: float,
        c_ad: float,
        c_aa: float,
        correlated: bool = True
) -> float:
    return qber_from_coincidences(
        c_00=c_dd,
        c_01=c_da,
        c_10=c_ad,
        c_11=c_aa,
        correlated=correlated
    )

def qy(
        c_rr: float,
        c_rl: float,
        c_lr: float,
        c_ll: float,
        correlated: bool = True
) -> float:
    return qber_from_coincidences(
        c_00=c_rr,
        c_01=c_rl,
        c_10=c_lr,
        c_11=c_ll,
        correlated=correlated
    )