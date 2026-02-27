import math

from .params import DiskParams

def Vg_cm_s(cs_cm_s: float, params: DiskParams) -> float:
    if params.alpha < 0.0:
        raise ValueError("alpha must be nonnegative.")
    if cs_cm_s <= 0.0:
        raise ValueError("cs_cm_s must be positive.")
    return math.sqrt(params.alpha) * cs_cm_s

def turbulent_relative_velocity_cm_s(St_1: float, St_2: float, cs_cm_s: float, OmegaK: float, params: DiskParams) -> float:
    if St_1 < 0.0 or St_2 < 0.0:
        raise ValueError("Stoke's number must be nonnegative.")
    if params.reynold <= 0.0:
        raise ValueError("Reynold's number must be positive.")
    if OmegaK <= 0.0:
        raise ValueError("OmegaK must be positive.")
    if cs_cm_s <= 0.0:
        raise ValueError("cs must be positive.")
    if St_2 > St_1:
        St_1, St_2 = St_2, St_1

    inv_sqrt_reynold = 1 / math.sqrt(params.reynold)
    Vg = Vg_cm_s(params.alpha, cs_cm_s)

    # Very Small Particle Regime
    if St_1 < inv_sqrt_reynold and St_2 < inv_sqrt_reynold:
        turnover_time_s = 1 / OmegaK
        kolmogorov_s = turnover_time_s * inv_sqrt_reynold
        return math.sqrt((Vg ** 2) * (turnover_time_s / kolmogorov_s) * ((St_1 - St_2) ** 2))
    # Heavy Particle Regime
    elif St_1 >= 1.0 and St_2 >= 1.0:
        return math.sqrt((Vg ** 2) * (1 / (1 + St_1) + 1 / (1 + St_2)))
    elif St_1 >= 1.0 and St_2 < 1.0:
        return math.sqrt((Vg ** 2) * (1 / (1 + St_1) + St_2 / (1 + St_2)))
    # Intermediate Regime
    else:
        epsilon = St_2 / St_1 if St_1 > 0 else 0.0
        return math.sqrt((Vg ** 2) * (2 * params.y_a * (1 + epsilon) + (2 / (1 + epsilon)) * (1 / (1 + params.y_a) + (epsilon ** 3) / (params.y_a + epsilon))) * St_1)