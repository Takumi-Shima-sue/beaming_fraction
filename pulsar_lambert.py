import numpy as np
from scipy.special import lambertw

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.coordinates import Angle
import astropy.units as u
from matplotlib.patches import Polygon

def ra_dec_to_deg(ra_str, dec_str):
    """
    RA/Dec を文字列 (例: "17:45:44", "-29:00:00") から度に変換する。

    Parameters
    ----------
    ra_str : str
        RA in "hh:mm:ss"
    dec_str : str
        Dec in "±dd:mm:ss"

    Returns
    -------
    ra_deg, dec_deg : float
        RA, Dec in degrees
    """
    ra_deg = Angle(ra_str + " hours").degree
    dec_deg = Angle(dec_str + " degrees").degree
    return ra_deg, dec_deg

def xy(ra_deg, dec_deg, dis):
    """
    (RA, Dec, distance) -> (x, y) in Galactic coordinates.
    RA, Dec [deg], distance [kpc or chosen unit]

    Returns
    -------
    x, y : ndarray or float
        Projected galactic coordinates (x,y)
    """
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    gal = c.galactic

    l_rad = gal.l.radian
    b_rad = gal.b.radian
    dis = np.asarray(dis, dtype=float)

    x = dis * np.cos(l_rad) * np.cos(b_rad)
    y = dis * np.sin(l_rad) * np.cos(b_rad)

    return -x, -y   # 符号は元のコードに合わせてマイナス

def lb(ra_deg, dec_deg):
    """
    (RA, Dec) -> Galactic (l, b) in degrees.
    RA, Dec [deg]

    Returns
    -------
    l, b : float
        Galactic longitude, latitude [deg]
    """
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    gal = c.galactic
    return gal.l.deg, gal.b.deg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1) 文字列→数値（度）→ [-180,180] → ラジアン 変換ヘルパー ---
def lb_to_aitoff_rad(l_deg, b_deg):
    """
    l_deg, b_deg: array-like / pandas.Series / list
    文字列や '°' が混ざっていてもOK。NaNは落とします。
    戻り値: (l_rad, b_rad) どちらも numpy.ndarray
    """
    # Series/ndarray/list → Series（文字列でもOK）
    l = pd.Series(l_deg)
    b = pd.Series(b_deg)

    # '°'や空白を除去（文字列の場合のみ適用）
    if l.dtype == 'O':
        l = l.astype(str).str.replace('°', '', regex=False).str.strip()
    if b.dtype == 'O':
        b = b.astype(str).str.replace('°', '', regex=False).str.strip()

    # 数値化（文字列も数値へ）。ダメな値はNaNへ
    l = pd.to_numeric(l, errors='coerce')
    b = pd.to_numeric(b, errors='coerce')

    # ペアで有効な行だけに絞る
    mask = l.notna() & b.notna()
    l = l[mask].to_numpy(dtype=float)
    b = b[mask].to_numpy(dtype=float)

    # [0,360) → [-180,180]
    l = (np.remainder(l + 180.0, 360.0)) - 180.0

    # ラジアンへ
    return -np.deg2rad(l), np.deg2rad(b)

# --- 2) 例：あなたのデータ列をここに差し替え ---
# 例: snr_l, snr_b, pwn_l, pwn_b, unid_l, unid_b という Series/配列がある想定
# snr_l = ...
# snr_b = ...
# pwn_l = ...
# pw
def _l_to_x_rad_left(l_deg):
    # [0,360)->[-180,180] に包んでから符号反転
    l = (np.remainder(np.asarray(l_deg, float) + 180.0, 360.0)) - 180.0
    return -np.deg2rad(l)

def _b_to_y_rad(b_deg):
    return np.deg2rad(np.asarray(b_deg, float))

# ---- 矩形をPolygonで追加（0/360跨ぎは自動二分, 端はεだけ内側） ----
def add_lb_box_aitoff(
    ax, l_min, l_max, b_min, b_max, *,
    edgecolor='C0', facecolor='none', linewidth=2.0, linestyle='-',
    alpha=1.0, label=None, zorder=3, eps_deg=1e-6
):
    """
    Aitoff(左向き増加)で (l_min..l_max, b_min..b_max) を Polygon で描く。
    l 範囲が 0/360 を跨ぐ場合は [l_min..360-ε] と [0+ε..l_max] の 2ポリゴン。
    label は最初のポリゴンにのみ付与。
    """
    l_min = float(l_min) % 360.0
    l_max = float(l_max) % 360.0

    def _add_one(l1, l2, b1, b2, lab):
        # 端をεだけ内側に寄せる（±180°ど真ん中を避ける）
        l1a = np.clip(l1, 0.0 + eps_deg, 360.0 - eps_deg)
        l2a = np.clip(l2, 0.0 + eps_deg, 360.0 - eps_deg)

        # 反時計回り（下辺→右辺→上辺→左辺）に頂点を並べる
        L = np.array([l1a, l2a, l2a, l1a])
        B = np.array([b1,  b1,  b2,  b2 ])

        X = _l_to_x_rad_left(L)
        Y = _b_to_y_rad(B)
        poly = Polygon(
            np.column_stack([X, Y]),
            closed=True, facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder,
            label=lab
        )
        ax.add_patch(poly)

    if l_min > l_max:
        # 0/360 を跨ぐ → 2つに分割（縫い目を跨がない）
        _add_one(l_min, 360.0 - eps_deg, b_min, b_max, label)  # ラベルはココだけ
        _add_one(0.0 + eps_deg, l_max, b_min, b_max, None)
    else:
        _add_one(l_min, l_max, b_min, b_max, label)



R_NS = 1e6       # cm
M_NS = 1.4e33    # g
I    = 1e45       # g cm^2
c    = 2.99792458e10  # cm/s

# --- ここは既存コードをそのまま利用（B の統一減衰） ---
def B_decay(t, B0, tau_ohm, alpha_i):
    t = np.asarray(t, dtype=float)
    if alpha_i == 0:
        return B0 * np.exp(-t / tau_ohm)
    else:
        return np.clip(B0 * (1.0 + alpha_i * t / tau_ohm)**(-1.0/alpha_i), 0, None)

# --- ∫ B^2 dt の閉形式（指数 / 統一減衰の両対応） ---
def int_B2_closed(t, B0, tau_ohm, alpha_i):
    t  = np.asarray(t, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    if np.isclose(alpha_i, 0.0, atol=1e-12):
        return (B0**2) * (tau_ohm/2.0) * (1.0 - np.exp(-2.0*t/tau_ohm))
    a = alpha_i / tau_ohm
    p = 1.0 - 2.0/alpha_i
    if np.isclose(alpha_i, 2.0, atol=1e-12):
        return (B0**2 / a) * np.log1p(a * t)
    return (B0**2 / (a * p)) * ((1.0 + a*t)**p - 1.0)

# --- Lambert W を用いた α(t)（厳密・枝は -1） ---
def alpha_lambert_equalk(t, P0, alpha0, B0, tau_ohm, alpha_i, *, R_NS, I, c):

    t      = np.asarray(t, dtype=float)
    P0     = np.asarray(P0, dtype=float)
    alpha0 = np.asarray(alpha0, dtype=float)
    B0     = np.asarray(B0, dtype=float)

    s0, c0 = np.sin(alpha0), np.cos(alpha0)
    u0     = 1.0 / (s0**2)                     # u = 1/sin^2 α
    C0     = u0 - np.log(u0)                   # 初期の定数項
    Omega0 = 2.0*np.pi / P0

    # κ = (R^6 / I c^3) * Ω0^2 * (cos^4 α0 / sin^2 α0)
    kappa  = (R_NS**6 / (I * c**3)) * (Omega0**2) * (c0**4) / (s0**2)

    # S(t) = ∫ B^2 dt を閉形式で
    S      = int_B2_closed(t, B0, tau_ohm, alpha_i)

    # C(t) = 2 κ S(t) + C0
    C      = 2.0 * kappa * S + C0
    z      = -np.exp(-C)
    # α∈(0,π/2] → u≥1 → LambertW の枝は -1
    u      = -lambertw(z, k=-1).real
    u      = np.maximum(u, 1.0)                # 数値丸めのガード
    s      = 1.0 / np.sqrt(u)
    # 微小ゼロ割防止
    s      = np.clip(s, 1e-300, 1.0)
    alpha  = np.arcsin(s)
    return alpha

# --- 第一積分から P(t) を即時に（ODE 不要・厳密） ---
def P_from_alpha(alpha, P0, alpha0):
    """
    Ω cos^2 α / sin α = const → P(t) = P0 * (sin α0 * cos^2 α) / (cos^2 α0 * sin α)
    ベクトル化済み。
    """
    s0, c0 = np.sin(alpha0), np.cos(alpha0)
    s,  c  = np.sin(alpha),  np.cos(alpha)
    ratio  = (s0 * c**2) / (c0**2 * np.clip(s, 1e-300, 1.0))
    return P0 * ratio

# --- dP/dt（診断や Pdot サンプリングに） ---
def Pdot_from_P_alpha_B(P, alpha, B, *, R_NS, I, c):
    """
    k0=k1=k2=1 の場合の Pdot:
    dP/dt = (4π^2 R^6 / I c^3) * B^2 * (1 + sin^2 α) / P
    """
    pref = (4.0 * np.pi**2) * (R_NS**6) / (I * c**3)
    return pref * (B**2) * (1.0 + np.sin(alpha)**2) / P

# --- Lambert 版：サンプルを一気に進化させてフィルタまでかける ---
def simulate_and_store_lambert(
    size,
    P0_log10, P0_sigma_log10,
    B0_log10, B0_sigma_log10,
    t_max, tau_ohm, alpha_i,
    *,
    rng=np.random
):
    """
    あなたの ODE ベース関数と同じ入出力構造で、Lambert 版を提供。
    k0=k1=k2=1 を想定（厳密一致）。ベクトル化のみで超高速。
    """
    # サンプリング
    P0_list     = 10.0**rng.normal(P0_log10, P0_sigma_log10, size=size)
    B0_list     = 10.0**rng.normal(B0_log10, B0_sigma_log10, size=size)
    alpha0_list = np.arccos(rng.uniform(0.0, 1.0, size=size))  # 一様 cos 分布
    t_list      = rng.uniform(0.0, t_max, size=size)

    # α(t) と P(t)
    alpha_list  = alpha_lambert_equalk(
        t_list, P0_list, alpha0_list, B0_list, tau_ohm, alpha_i,
        R_NS=R_NS, I=I, c=c
    )
    P_list      = P_from_alpha(alpha_list, P0_list, alpha0_list)

    # B(t), Pdot(t)
    B_list      = B_decay(t_list, B0_list, tau_ohm, alpha_i)
    Pdot_list   = Pdot_from_P_alpha_B(P_list, alpha_list, B_list, R_NS=R_NS, I=I, c=c)

    # デスライン（あなたの式に合わせる）
    R6 = R_NS / 1e6
    def deathline_multipole(P, chi=1.0, R6=1.0):
        return 8.28e-19 * chi**(-0.5) * R6**(-4) * P**3

    mask = Pdot_list > deathline_multipole(P_list, chi=1.0, R6=R6)

    return {
        'P_samples':      P_list[mask],
        'alpha_samples':  alpha_list[mask],
        'P_dot_samples':  Pdot_list[mask],
        'B_samples':      B_list[mask],
        't_samples':      t_list[mask],
        'P_0_list':       P0_list[mask],
        'B_0_list':       B0_list[mask],
        'alpha_0_list':   alpha0_list[mask],
        'P_dot_0_list':  Pdot_from_P_alpha_B(P0_list[mask], alpha0_list[mask], B0_list[mask], R_NS=R_NS, I=I, c=c)
    }


import numpy as np
from scipy.special import lambertw

# =========================================
# Constants (cgs)
# =========================================
YEAR = 365.25 * 86400.0  # [s]
c    = 2.99792458e10     # [cm/s]
R_NS = 1.0e6             # [cm] ~ 10 km
I    = 1.36e45           # [g cm^2]

# =========================================
# Helpers
# =========================================
def _to_1d(x):
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return a.reshape(1), True, ()
    return a.ravel(), False, a.shape


# ============================================================
# 3-stage B-field (pairwise)
# ============================================================
def B_field_pairwise(
    t, B0, tau_late_sec, a_late,
    A1_yr=1e14, b1=-0.8, A2_yr=6e8, b2=-0.2,
    a1=-0.13, a2=-3.0
):
    """
    Appendix A の (A1)-(A3) を pairwise で実装。
    - tau_late_sec は [s]。A1_yr, A2_yr は [yr]（内部で秒に換算）。
    - a_late の符号は「そのまま」。負値にすれば後期で減衰。
    """
    t1, t_scalar, tshape = _to_1d(t)
    B1, B_scalar, Bshape = _to_1d(B0)

    # pairwise 揃え
    if not t_scalar and not B_scalar and (t1.size != B1.size):
        raise ValueError(f"pairwise: len(t)={t1.size} != len(B0)={B1.size}")
    if t_scalar and not B_scalar:
        t1 = np.full_like(B1, t1[0], dtype=float)
    if not t_scalar and B_scalar:
        B1 = np.full_like(t1, B1[0], dtype=float)

    n = t1.size
    out = np.empty(n, dtype=float)

    tauL = max(float(tau_late_sec), 1e-30)
    tau1 = np.maximum(A1_yr * (B1**b1) * YEAR, 1e-30)
    tau2 = np.maximum(A2_yr * (B1**b2) * YEAR, 1e-30)

    A1m = (tau2 > tau1) & (tau2 < tauL)        # tau1 < tau2 < tau_late
    A2m = (tau1 < tauL) & (tauL < tau2)        # tau1 < tau_late < tau2
    A3m = (tauL < tau1) & (tau1 < tau2)        # tau_late < tau1 < tau2
    Em  = ~(A1m | A2m | A3m)                   # fallback: A2型

    if np.any(A1m):
        i = A1m
        out[i] = (B1[i]
                  * (1.0 + t1[i]/tau1[i])**(a1)
                  * (1.0 + t1[i]/tau2[i])**(a2 - a1)
                  * (1.0 + t1[i]/tauL   )**(a_late - a2))
    if np.any(A2m):
        i = A2m
        out[i] = (B1[i]
                  * (1.0 + t1[i]/tau1[i])**(a1)
                  * (1.0 + t1[i]/tauL   )**(a_late - a1))
    if np.any(A3m):
        i = A3m
        out[i] = B1[i] * (1.0 + t1[i]/tauL)**(a_late)
    if np.any(Em):
        i = Em
        out[i] = (B1[i]
                  * (1.0 + t1[i]/tau1[i])**(a1)
                  * (1.0 + t1[i]/tauL   )**(a_late - a1))

    # 形に戻す
    if t_scalar and B_scalar:
        return float(out[0])
    if not t_scalar and B_scalar:
        return out.reshape(tshape)
    if t_scalar and not B_scalar:
        return out.reshape(Bshape)
    return out


# ============================================================
# Fast approximate ∫_0^t B^2 dt (pairwise, log-time + Simpson)
# ============================================================
def integral_B2_pairwise_fast_np(t, B0, tau_late_sec, a_late, M=33):
    t1, t_scalar, tshape = _to_1d(t)
    B1, B_scalar, Bshape = _to_1d(B0)

    if not t_scalar and not B_scalar and (t1.size != B1.size):
        raise ValueError(f"pairwise: len(t)={t1.size} != len(B0)={B1.size}")
    if t_scalar and not B_scalar:
        t1 = np.full_like(B1, t1[0], dtype=float)
    if not t_scalar and B_scalar:
        B1 = np.full_like(t1, B1[0], dtype=float)

    n = t1.size
    out = np.zeros(n, dtype=float)
    tauL = float(tau_late_sec) if tau_late_sec > 0.0 else 1e-30

    # Simpson weights
    M = int(M)
    if M % 2 == 0: M += 1
    w = np.ones(M)
    if M >= 3:
        w[1:M-1:2] = 4.0
        w[2:M-1:2] = 2.0

    for i in range(n):
        ti, b0 = float(t1[i]), float(B1[i])
        if (ti <= 0.0) or (b0 <= 0.0):
            out[i] = 0.0
            continue
        S  = np.log1p(ti / tauL)   # s ∈ [0,S]
        s  = np.linspace(0.0, S, M)
        es = np.exp(s)
        u  = tauL * (es - 1.0)     # u ∈ [0,ti]

        Bu = B_field_pairwise(u, b0, tauL, a_late)
        f  = (Bu * Bu) * tauL * es # f(s) = B^2 * du/ds

        ds = S / (M - 1.0)
        out[i] = (w @ f) * ds / 3.0

    if t_scalar and B_scalar: return float(out[0])
    if not t_scalar and B_scalar: return out.reshape(tshape)
    if t_scalar and not B_scalar: return out.reshape(Bshape)
    return out

def _u_from_C_monotone(C):
    C = np.asarray(C, dtype=float)
    C = np.maximum(C, 1.0)
    u = np.empty_like(C)

    delta   = C - 1.0
    mask_bp = (delta < 1e-30)
    mask_lg = (C > 60.0)
    mask_md = ~(mask_bp | mask_lg)

    if np.any(mask_bp):
        d = delta[mask_bp]
        u[mask_bp] = 1.0 + np.sqrt(2.0*d) * (1.0 + d/3.0)
    if np.any(mask_md):
        Cm = C[mask_md]
        z  = -np.exp(-Cm)
        z  = np.clip(z, -np.e**-1 + 1e-16, -1e-300)
        u[mask_md] = -lambertw(z, k=-1).real
    if np.any(mask_lg):
        Cl = C[mask_lg]
        u[mask_lg] = Cl + np.log(Cl)

    for _ in range(10):
        u  = np.maximum(u, 1.0 + 1e-14)
        f  = u - np.log(u) - C
        df = 1.0 - 1.0/u
        u -= f/df

    return np.maximum(u, 1.0)


def _alpha_lambert_3stage(
    t_sec, P0, alpha0, B0,
    tau_late_sec, a_late,
    A1_yr=1e14, b1=-0.8, A2_yr=6e8, b2=-0.2,
    a1=-0.13, a2=-3.0, M=33
):
    t_sec  = np.asarray(t_sec,  dtype=float)
    P0     = np.asarray(P0,     dtype=float)
    alpha0 = np.asarray(alpha0, dtype=float)
    B0     = np.asarray(B0,     dtype=float)

    s0, c0 = np.sin(alpha0), np.cos(alpha0)
    u0     = 1.0 / np.maximum(s0**2, 1e-300)
    C0     = u0 - np.log(u0)
    Omega0 = 2.0*np.pi / P0

    # κ = (R^6 / (I c^3)) * Ω0^2 / 4 * (cos^4 α0 / sin^2 α0)
    kappa  = (R_NS**6) / (I * c**3) * (Omega0**2) / 4.0 * (c0**4) / np.maximum(s0**2, 1e-300)

    # ∫B^2 dt
    S = integral_B2_pairwise_fast_np(t_sec, B0, tau_late_sec, a_late, M=M)

    C = 2.0 * kappa * S + C0
    u = _u_from_C_monotone(C)
    s = 1.0 / np.sqrt(u)   # = sin α
    return s


def _P_from_alpha(s, P0, alpha0):
    s0, c0 = np.sin(alpha0), np.cos(alpha0)
    c = np.sqrt(np.maximum(1.0 - s**2, 0.0))
    ratio  = (s0 * c**2) / (np.maximum(c0**2, 1e-300) * np.clip(s, 1e-300, 1.0))
    return P0 * ratio


def _Pdot_from_P_alpha_B(P, s, B):
    # 係数を 4π^2 に統一（初期値生成と整合）
    pref = (4.0 * np.pi**2) * (R_NS**6) / (I * c**3)
    return pref * (B**2) * (1.0 + s**2) / P


def deathline_multipole(P, chi=1.0, R6=1.0):
    return 8.28e-19 * chi**(-0.5) * R6**(-4) * P**3


# ============================================================
# Public: simulate with the 3-stage model consistently
# ============================================================
def simulate_and_store_lambert_vigano(
    k0, k1, k2, size,
    P_log, P_sig, B_log, B_sig,
    t_max_yr,              # [yr]
    tau_late_yr, a_late,   # [yr], 典型的に負
    A1_yr=1e14, b1=-0.8, A2_yr=6e8, b2=-0.2,
    a1=-0.13, a2=-3.0,
    M=33
):
    if not (np.isclose(k0,1.0) and np.isclose(k1,1.0) and np.isclose(k2,1.0)):
        raise ValueError("Lambert 実装は k0=k1=k2=1 の場合に正確です。")

    rng = np.random

    # sampling
    P_0_list     = 10.0**rng.normal(P_log, P_sig, size=size)         # [s]
    B_0_list     = 10.0**rng.normal(B_log, B_sig, size=size)         # [G]
    alpha_0_list = np.arccos(rng.uniform(0.0, 1.0, size=size))       # [rad]
    P_dot_0_list = (4.0*np.pi**2) * (R_NS**6)/(I*c**3) * (B_0_list**2) * (1.0 + np.sin(alpha_0_list)**2) / P_0_list

    # 年→秒
    t_samples    = rng.uniform(0.0, t_max_yr, size=size) * YEAR      # [s]
    tau_late_sec = tau_late_yr * YEAR                                # [s]

    # α(t), P(t)
    s_final = _alpha_lambert_3stage(
        t_samples, P_0_list, alpha_0_list, B_0_list,
        tau_late_sec, a_late,
        A1_yr=A1_yr, b1=b1, A2_yr=A2_yr, b2=b2, a1=a1, a2=a2, M=M
    )
    alpha_final = np.arcsin(np.clip(s_final, 1e-300, 1.0))
    P_final     = _P_from_alpha(s_final, P_0_list, alpha_0_list)

    # B(t), Ṗ(t)
    B_samples     = B_field_pairwise(
        t_samples, B_0_list, tau_late_sec, a_late,
        A1_yr=A1_yr, b1=b1, A2_yr=A2_yr, b2=b2, a1=a1, a2=a2
    )
    P_dot_samples = _Pdot_from_P_alpha_B(P_final, s_final, B_samples)

    # death line マスク
    R6   = R_NS / 1.0e6
    mask = (P_dot_samples > deathline_multipole(P_final, chi=1.0, R6=R6)) & (P_final > 3e-3)

    return {
        'P_samples':      P_final[mask],
        'alpha_samples':  alpha_final[mask],
        'P_dot_samples':  P_dot_samples[mask],
        'B_samples':      B_samples[mask],
        't_samples':      t_samples[mask],
        'P_0_list':       P_0_list[mask],
        'B_0_list':       B_0_list[mask],
        'P_dot_0_list':   P_dot_0_list[mask],
        'alpha_0_list':   alpha_0_list[mask],
        'mask':           mask
    }


# -*- coding: utf-8 -*-
"""
Milky Way axisymmetric potential (disk + bulge + halo) and utilities.
Units: length [kpc], velocity [km/s], mass [Msun], time [kpc / (km/s)].
Potential in [(km/s)^2].
"""

import numpy as np
from numba import njit

R_c = 8.5

km_per_kpc = 3.24077929e-14  # [kpc/km]


import numpy as np

# Maxwell 分布
def f_maxwell(v, s):
    return np.sqrt(2/np.pi) * v**2/s**3 * np.exp(-v**2/(2*s**2))

# 混合分布からサンプリング
def sample_mixture(n, w, s1, s2):
    n1 = np.random.binomial(n, w)   # σ1から引く個数
    v1 = np.random.normal(scale=s1, size=(3, n1))
    v2 = np.random.normal(scale=s2, size=(3, n-n1))
    return np.linalg.norm(np.hstack([v1, v2]), axis=0)


# FGK06 A5 パラメータ
a, b = 1.64, 4.01
R0, R1 = 8.5, 0.55  # kpc

def surf_density(rho):
    """面密度 Σ(ρ)（正規化前）。指数の分母は (R0+R1) が正しい。"""
    rho = np.asarray(rho, dtype=float)
    y = ((rho + R1) / (R0 + R1))**a * np.exp(-b * (rho - R0) / (R0 + R1))
    y[rho <= 0] = 0.0
    return y

def radial_pdf(rho):
    """平面上の位置サンプリング用 PDF ∝ ρ Σ(ρ)"""
    return rho * surf_density(rho)

def precompute_rho_cdf(rho_max=50.0, ngrid=20001):
    rho_grid = np.linspace(1e-3, rho_max, ngrid)  # 0は避ける
    pdf = radial_pdf(rho_grid)
    pdf[np.isnan(pdf)] = 0.0
    # 台形則で CDF
    cdf = np.cumsum((pdf[1:] + pdf[:-1]) * np.diff(rho_grid) / 2.0)
    cdf = np.concatenate([[0.0], cdf])
    cdf /= cdf[-1]
    cdf = np.maximum.accumulate(cdf)
    cdf[-1] = 1.0
    return rho_grid, cdf

def sample_rho_fast(n, rho_grid, cdf):
    u = np.random.random(n)
    return np.interp(u, cdf, rho_grid)

def sample_z(n, z0=0.05):
    """
    z方向の指数分布サンプルを生成。
    - z0: スケールハイト（kpc）
    - n: サンプル数
    """
    u = np.random.uniform(-1, 1, n)  # 一様分布 [-1, 1]
    return -z0 * np.sign(u) * np.log(1 - np.abs(u))

_rng = np.random.default_rng()

params = [
    (4.25, 3.48, 1.57),
    (4.25, 3.48, 4.71),
    (4.89, 4.90, 4.09),
    (4.89, 4.90, 0.95),
]


def theta_r_xy(r, k, r0, theta0):
    r = np.asarray(r, dtype=float)
    return -k * np.log(r / r0) - theta0

def spiral_thetas(r_samples, params=params):
    # shape: (N, n_arms)
    return np.stack([theta_r_xy(r_samples, *p) for p in params], axis=1)


def sample_theta_corr(r_base):
    r_base = np.asarray(r_base, dtype=float)
    theta_corr = _rng.uniform(0.0, 2 * np.pi, size=r_base.shape)
    sigma_theta = theta_corr * np.exp(-0.35 * r_base)
    return _rng.normal(0.0, sigma_theta, size=r_base.shape)




def add_scatter(rho_raw, theta_arm, sigma_frac=0.07, rng=None):
    """
    rho_raw, theta_arm: arrays of same shape
    sigma_frac: σ/ρ の比率（デフォルト 0.07）
    """
    if rng is None:
        rng = np.random.default_rng()

    rho_raw = np.asarray(rho_raw, float)
    theta_arm = np.asarray(theta_arm, float)

    # 基準点 (x0,y0)
    x0 = rho_raw * np.cos(theta_arm)
    y0 = rho_raw * np.sin(theta_arm)

    # 2D Gaussian scatter
    sigma = sigma_frac * rho_raw
    dx = rng.normal(0, sigma)
    dy = rng.normal(0, sigma)

    # 新しい点
    x = x0 + dx
    y = y0 + dy

    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta = np.mod(theta, 2*np.pi)
    return rho, theta

def choose_random_arm_theta(theta_array):
    """
    theta_array: shape (N, 4) — 各rhoに対する4本のアーム候補角度
    戻り値: shape (N,) — 各行からランダムに選ばれた1つの角度
    """
    N, n_arms = theta_array.shape
    indices = np.random.randint(0, n_arms, size=N)  # 各行で選ぶインデックス
    return theta_array[np.arange(N), indices]

import numpy as np

def sample_z(n, z0=0.05):
    """
    z方向の指数分布からサンプル
    - 平面対称な |z| 指数分布（PDF ∝ exp(-|z|/z0)）
    """
    u = np.random.uniform(-1, 1, n)
    return -z0 * np.sign(u) * np.log(1 - np.abs(u))

def choose_random_arm_theta(theta_arms):
    """
    各スパイラルアーム候補のうち1つをランダムに選択
    - theta_arms.shape = (n_samples, 4)
    """
    n = theta_arms.shape[0]
    arm_idx = np.random.randint(0, 4, n)
    return theta_arms[np.arange(n), arm_idx]

rho_grid, cdf = precompute_rho_cdf(rho_max=50.0, ngrid=40001)

def sample_maxwellian(n, sigma):
    """
    f_max(v | sigma) に従う速度ベクトルの大きさ v をサンプリング。
    sigma: 標準偏差
    n: サンプル数
    """
    # 各成分を独立な正規分布からサンプリング
    vx = np.random.normal(0, sigma, size=n)
    vy = np.random.normal(0, sigma, size=n)
    vz = np.random.normal(0, sigma, size=n)
    return np.sqrt(vx**2 + vy**2 + vz**2)

def sample_mixture_maxwellian(n, w, sigma1, sigma2):
    """
    2成分混合Maxwell分布からの速度vのサンプリング
    w: sigma1成分の重み（確率）
    sigma1, sigma2: それぞれのMaxwell分布の標準偏差
    """
    # どちらの分布からサンプルするかを決定
    choice = np.random.uniform(0, 1, size=n) < w

    v_samples = np.empty(n)
    n1 = np.sum(choice)
    n2 = n - n1

    # 対応するσの分布からサンプリング
    v_samples[choice] = sample_maxwellian(n1, sigma1)
    v_samples[~choice] = sample_maxwellian(n2, sigma2)
    return v_samples

def sample_galactic_coordinates(n):
    """
    銀河内の位置 (x, y, z) を一括サンプリング
    """
    # --- r のサンプリング ---
    u = np.random.uniform(0, 1, n)
    rho_samples = np.interp(u, cdf, rho_grid)

    # --- θ: スパイラルアーム候補から選択 ---
    theta_arms = spiral_thetas(rho_samples)          # (n, 4)
    theta_samples = sample_theta_corr(rho_samples)   # scatter (n,)
    theta_main = choose_random_arm_theta(theta_arms) # main arm θ (n,)
    theta_final = theta_main + theta_samples
    rho_samples, theta_final = add_scatter(rho_samples, theta_final)

    # --- z サンプリング ---
    z0=0.05  
    z_samples = sample_z(n, z0=z0)

    # --- x, y, z 座標変換 ---
    x = rho_samples * np.cos(theta_final)
    y = rho_samples * np.sin(theta_final)
    z = z_samples
    
    # --- v サンプリング ---
    v = sample_mixture_maxwellian(n, w=0.4, sigma1=90.0, sigma2=500.0)

    return {
        "r_samples": rho_samples,
        "z_samples": z,
        "theta_samples": theta_final,
        "x_samples": x,
        "y_samples": y,
        "v_samples": v
    }


# polar plot
def plot_pulsar_positions(rho_samples, theta_samples):


    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")

    ax.scatter(theta_samples, rho_samples, s=1, alpha=0.5, color="blue")
    ax.set_theta_direction(-1)        # θを時計回りに
    ax.set_rlim(0, 20)       # 半径の上限

    ax.set_title("Monte Carlo Initial Pulsar Distribution", va="bottom")
    plt.show()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P(t), alpha(t) の数値積分（solve_ivp）比較プロッタ
- Ohmic 減衰
- Hall 風の統一減衰（B(t) = B0 (1 + α_i f_B B0^{α_i} t)^(-1/α_i)）
- ConstB（B一定）
- Vacuum（真空双極子：k0=0, k1=2/3, k2=2/3）

使い方:
  python pulsar_spin_models.py
  （オプション）観測点を重ねたい場合は `psr_P`, `psr_P_d` を numpy 配列で下の MAIN 部に渡す。
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from typing import Optional

# -----------------------------
# 物理定数（cgs）
# -----------------------------
R_NS = 1.0e6   # 中性子星半径 [cm] (10 km)
I     = 1.0e45 # 慣性モーメント [g cm^2]
c     = 3.0e10 # 光速 [cm/s]

# スピンダウン係数（力学係数）
# 一般力学形: dP/dt ∝ B^2 (k0 + k1 sin^2 α) / P
#             dα/dt ∝ -B^2 k2 sin α cos α / P^2
k_0, k_1, k_2 = 1.0, 1.2, 1.0   # Force-free に寄せた例（任意）

# 便利なプレファクター
PREF = 4.0*np.pi**2 * R_NS**6 / (I * c**3)  # [s^(-1) G^(-2) s] の次元を打ち消す

# -----------------------------
# B(t) モデル
# -----------------------------
def B_ohm(t: np.ndarray | float, B0: float, tau_ohm: float) -> np.ndarray:
    """指数（Ohmic）減衰: B(t) = B0 * exp(-t / tau_ohm)"""
    return B0 * np.exp(-np.asarray(t, dtype=float) / tau_ohm)

def B_hall_unified(t: np.ndarray | float, B0: float, f_B: float, alpha_i: float) -> np.ndarray:
    """
    Hall 風の統一減衰:
      B(t) = B0 * (1 + α_i f_B B0^{α_i} t)^(-1/α_i)
    alpha_i -> 0 の極限は指数減衰に一致（log 極限）。
    """
    t = np.asarray(t, dtype=float)
    if np.isclose(alpha_i, 0.0, atol=1e-14):
        # 近似的に指数形へ（f_B B0^{α_i} ~ f_B とみなす）
        return B0 * np.exp(-f_B * t)
    a = alpha_i * f_B * (B0**alpha_i)
    return np.clip(B0 * (1.0 + a * t)**(-1.0/alpha_i), 0.0, None)

# -----------------------------
# ODE（一般形）
# -----------------------------
def ode_general(t: float, y: np.ndarray, B_t: float, k0: float, k1: float, k2: float) -> list[float]:
    """
    y = [P, alpha]
    dP/dt = PREF * B(t)^2 * (k0 + k1 sin^2 α) / P
    dα/dt = -PREF * B(t)^2 * k2 * sin α cos α / P^2
    """
    P, alpha = float(y[0]), float(y[1])
    # 数値安定化
    P     = max(P, 1e-9)
    alpha = np.clip(alpha, 1e-12, np.pi/2 - 1e-12)

    s, csa = np.sin(alpha), np.cos(alpha)
    dP_dt     = PREF * (B_t**2) * (k0 + k1 * s**2) / P
    dalpha_dt = -PREF * (B_t**2) * (k2 * s * csa) / (P**2)
    return [dP_dt, dalpha_dt]

# -----------------------------
# モデル別 ODE ラッパ
# -----------------------------
def ode_ohm(t, y, B0, tau_ohm, k0, k1, k2):
    B_t = float(B_ohm(t, B0, tau_ohm))
    return ode_general(t, y, B_t, k0, k1, k2)

def ode_hall(t, y, B0, f_B, alpha_i, k0, k1, k2):
    B_t = float(B_hall_unified(t, B0, f_B, alpha_i))
    return ode_general(t, y, B_t, k0, k1, k2)

def ode_constB(t, y, B0, k0, k1, k2):
    B_t = float(B0)
    return ode_general(t, y, B_t, k0, k1, k2)

def ode_vacuum(t, y, B0):
    """
    真空双極子: k0=0, k1=2/3, k2=2/3, B は一定（定義上の近似）
    """
    return ode_constB(t, y, B0, k0=0.0, k1=2.0/3.0, k2=2.0/3.0)

# -----------------------------
# 実行関数
# -----------------------------
def run_all_models(
    P0_list: list[float],
    alpha0_list: list[float],
    B0_list: list[float],
    *,
    # 時間設定
    years_min: float = 1e0,    # 出力の最小年（0年は log 取れないので 1年から）
    years_max: float = 1e8,    # 出力の最大年
    n_time: int = 2000,
    # 減衰パラメータ
    tau_ohm_years: float = 8.31e6,  # [yr]
    hall_alpha_i: float = 1.0,
    hall_f_B: float = 10.0**(-26.8),
    # 観測データ（任意）
    psr_P: Optional[np.ndarray] = None,
    psr_Pdot: Optional[np.ndarray] = None,
    # 保存先（任意）
    output_dir: Optional[str] = None
):
    """
    5本の B0 軌道を、Ohm / Hall / ConstB / Vacuum の4モデルで解いて比較プロット。
    """
    # 時間グリッド（秒）
    t_eval_years = np.logspace(np.log10(years_min), np.log10(years_max), n_time)
    t_eval = t_eval_years * (365.25 * 24.0 * 3600.0)
    t_span = (0.0, float(t_eval[-1]))

    tau_ohm = tau_ohm_years * (365.25 * 24.0 * 3600.0)

    # 結果を格納
    results_ohm   = []
    results_hall  = []
    results_const = []
    results_vac   = []

    # 積分
    for P0, a0, B0 in zip(P0_list, alpha0_list, B0_list):
        y0 = [float(P0), float(a0)]

        # Ohm
        sol_ohm = solve_ivp(
            ode_ohm, t_span, y0, t_eval=t_eval,
            args=(B0, tau_ohm, k_0, k_1, k_2), rtol=1e-8, atol=1e-12
        )
        B_ohm_arr  = B_ohm(t_eval, B0, tau_ohm)
        Pdot_ohm   = PREF * B_ohm_arr**2 / sol_ohm.y[0] * (k_0 + k_1 * np.sin(sol_ohm.y[1])**2)
        results_ohm.append((P0, a0, B0, sol_ohm, Pdot_ohm))

        # Hall-like unified
        sol_hall = solve_ivp(
            ode_hall, t_span, y0, t_eval=t_eval,
            args=(B0, hall_f_B, hall_alpha_i, k_0, k_1, k_2), rtol=1e-8, atol=1e-12
        )
        B_h_arr   = B_hall_unified(t_eval, B0, hall_f_B, hall_alpha_i)
        Pdot_hall = PREF * B_h_arr**2 / sol_hall.y[0] * (k_0 + k_1 * np.sin(sol_hall.y[1])**2)
        results_hall.append((P0, a0, B0, sol_hall, Pdot_hall))

        # ConstB
        sol_const = solve_ivp(
            ode_constB, t_span, y0, t_eval=t_eval,
            args=(B0, k_0, k_1, k_2), rtol=1e-8, atol=1e-12
        )
        Pdot_const = PREF * (B0**2) / sol_const.y[0] * (k_0 + k_1 * np.sin(sol_const.y[1])**2)
        results_const.append((P0, a0, B0, sol_const, Pdot_const))

        # Vacuum (k0=0,k1=k2=2/3, B一定)
        sol_vac = solve_ivp(
            ode_vacuum, t_span, y0, t_eval=t_eval,
            args=(B0,), rtol=1e-8, atol=1e-12
        )
        Pdot_vac = PREF * (B0**2) / sol_vac.y[0] * ((2.0/3.0) * np.sin(sol_vac.y[1])**2)
        results_vac.append((P0, a0, B0, sol_vac, Pdot_vac))

    # -----------------------------
    # プロット (P, Pdot, B, alpha)
    # -----------------------------
    fig, axs = plt.subplots(1, 4, figsize=(20, 8))

    # B0 色づけ
    B0_arr = np.array(B0_list, dtype=float)
    norm   = mcolors.Normalize(vmin=np.log10(B0_arr.min()), vmax=np.log10(B0_arr.max()))
    cmap   = cm.viridis

    linestyles = {'Ohm': '-', 'Hall': '--', 'ConstB': ':', 'Vacuum': '-.'}
    def c_from_B(B0): return cmap(norm(np.log10(B0)))

    # モデルごとに描画
    for label, res in [('Ohm', results_ohm), ('Hall', results_hall),
                       ('ConstB', results_const), ('Vacuum', results_vac)]:
        for P0, a0, B0, sol, Pdot in res:
            color = c_from_B(B0)
            t_yr  = sol.t / (365.25 * 24.0 * 3600.0)

            # P(t)
            axs[0].plot(t_yr, sol.y[0], linestyle=linestyles[label], color=color,
                        label=f"$B_0={B0:.1e}$" if label == 'Ohm' else None)

            # Pdot(t)
            axs[1].plot(t_yr, Pdot, linestyle=linestyles[label], color=color)

            # B(t)
            if label == 'Ohm':
                B_arr = B_ohm(sol.t, B0, tau_ohm)
            elif label == 'Hall':
                B_arr = B_hall_unified(sol.t, B0, hall_f_B, hall_alpha_i)
            else:
                B_arr = B0 * np.ones_like(sol.t)
            axs[2].plot(t_yr, B_arr, linestyle=linestyles[label], color=color)

            # alpha(t) [deg]
            axs[3].plot(t_yr, np.degrees(sol.y[1]), linestyle=linestyles[label], color=color)

    # 軸スケール・ラベル
    labels = ["P (s)", r"$\dot{P}$ (s s$^{-1}$)", "B (G)", r"$\alpha$ (deg)"]
    for ax, yl in zip(axs, labels):
        ax.set_xscale('log')
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(yl)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')

    # 凡例（B0 色凡例）
    handles, labels_ = axs[0].get_legend_handles_labels()
    if handles:
        axs[0].legend(handles=handles, labels=labels_, title="$B_0$ (G)",
                      bbox_to_anchor=(-0.8, 1), loc='upper left', frameon=False)

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f"{output_dir}/pulsar_simulation_P_Pdot_B_alpha.pdf", bbox_inches='tight')
    plt.show()

    # -----------------------------
    # P–Pdot 図
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    for label, res in [('Ohm', results_ohm), ('Hall', results_hall),
                       ('ConstB', results_const), ('Vacuum', results_vac)]:
        for _, _, B0, sol, Pdot in res:
            color = c_from_B(B0)
            ax.plot(sol.y[0], Pdot, linestyle=linestyles[label], color=color)

    # 観測点（任意）
    if (psr_P is not None) and (psr_Pdot is not None):
        ax.scatter(psr_P, psr_Pdot, s=10, alpha=0.5, label='Observed')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e-20, 1e-9)
    ax.set_xlabel("P (s)")
    ax.set_ylabel(r"$\dot{P}$ (s s$^{-1}$)")
    ax.grid(True, which="both", ls="-", alpha=0.3)

    # カラーバー
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$\log_{10} B_0$ (G)')

    # 線種の凡例
    ohm_line    = mlines.Line2D([], [], color='gray', linestyle='-',  label='Ohm')
    hall_line   = mlines.Line2D([], [], color='gray', linestyle='--', label='Hall')
    constb_line = mlines.Line2D([], [], color='gray', linestyle=':',  label='ConstB')
    vac_line    = mlines.Line2D([], [], color='gray', linestyle='-.', label='Vacuum')

    legend_items = [ohm_line, hall_line, constb_line, vac_line]
    if (psr_P is not None) and (psr_Pdot is not None):
        obs_pt = mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='Observed')
        legend_items.append(obs_pt)

    ax.legend(handles=legend_items, loc='upper left', frameon=False)

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f"{output_dir}/pulsar_simulation_PPdot_tracks.pdf", bbox_inches='tight')
    plt.show()


import numpy as np

KPC_TO_CM = 3.086e21
TEV_TO_ERG = 1.602

def _as_array(x):
    """None/スカラー/ndarrayのいずれでも ndarray にする（コピーしない）"""
    return np.asarray(x)
import numpy as np

KPC_TO_CM = 3.086e21
TEV_TO_ERG = 1.602

def _as_array(x):
    return np.asarray(x)

def pl_lumi_mc(
    f0, f0_err, gamma, gamma_err, d_kpc, d_err,
    E0=1.0, Emin=1.0, Emax=10.0,
    nsim=20000, random_state=None
):
    rng = np.random.default_rng(random_state)

    # 共同形状 S
    S = np.broadcast_shapes(
        _as_array(f0).shape, _as_array(f0_err).shape,
        _as_array(gamma).shape, _as_array(gamma_err).shape,
        _as_array(d_kpc).shape, _as_array(d_err).shape,
        _as_array(E0).shape, _as_array(Emin).shape, _as_array(Emax).shape,
    )

    # ブロードキャストして float 化
    f0      = np.broadcast_to(_as_array(f0),      S).astype(float)
    f0_err  = np.broadcast_to(_as_array(f0_err),  S).astype(float)
    gamma   = np.broadcast_to(_as_array(gamma),   S).astype(float)
    gamma_err = np.broadcast_to(_as_array(gamma_err), S).astype(float)
    d_kpc   = np.broadcast_to(_as_array(d_kpc),   S).astype(float)
    d_err   = np.broadcast_to(_as_array(d_err),   S).astype(float)
    E0      = np.broadcast_to(_as_array(E0),      S).astype(float)
    Emin    = np.broadcast_to(_as_array(Emin),    S).astype(float)
    Emax    = np.broadcast_to(_as_array(Emax),    S).astype(float)

    # サンプル（MC軸を先頭に）
    size = (nsim,) + S
    f0_sig = np.nan_to_num(f0_err,   nan=0.0, posinf=0.0, neginf=0.0)
    g_sig  = np.nan_to_num(gamma_err,nan=0.0, posinf=0.0, neginf=0.0)
    d_sig  = np.nan_to_num(d_err,    nan=0.0, posinf=0.0, neginf=0.0)

    f0_s = rng.normal(loc=f0,    scale=f0_sig, size=size)
    g_s  = rng.normal(loc=gamma, scale=g_sig,  size=size)
    d_s  = rng.normal(loc=d_kpc, scale=d_sig,  size=size)

    # E0/Emin/Emax を (nsim,)+S に拡張（ここが重要）
    E0_s   = np.broadcast_to(E0,   S)[None, ...]
    Emin_s = np.broadcast_to(Emin, S)[None, ...]
    Emax_s = np.broadcast_to(Emax, S)[None, ...]
    E0_s   = np.broadcast_to(E0_s,   f0_s.shape)
    Emin_s = np.broadcast_to(Emin_s, f0_s.shape)
    Emax_s = np.broadcast_to(Emax_s, f0_s.shape)

    # 積分（Γ=2 は log、Γ≠2 は解析式）— 全要素一括で np.where
    with np.errstate(divide='ignore', invalid='ignore'):
        mask_eq2 = np.isclose(g_s, 2.0, rtol=1e-12, atol=1e-12)
        # Γ≠2 の項
        integral_ne2 = (np.power(Emax_s, 2.0 - g_s) - np.power(Emin_s, 2.0 - g_s)) / (2.0 - g_s)
        # Γ=2 の項
        integral_eq2 = np.log(Emax_s / Emin_s)
        integral = np.where(mask_eq2, integral_eq2, integral_ne2)

    # 光度
    d_cm_s = d_s * KPC_TO_CM
    L_s = 4.0 * np.pi * (d_cm_s**2) * f0_s * np.power(E0_s, g_s) * integral * TEV_TO_ERG

    # 物理的に不正なサンプルを除外
    bad_bounds = ~np.isfinite(Emin_s) | ~np.isfinite(Emax_s) | (Emin_s <= 0) | (Emax_s <= Emin_s)
    bad = (f0_s <= 0) | (d_s <= 0) | ~np.isfinite(L_s) | bad_bounds
    L_s[bad] = np.nan

    mean   = np.nanmean(L_s, axis=0)
    q16    = np.nanpercentile(L_s, 16, axis=0)
    q84    = np.nanpercentile(L_s, 84, axis=0)
    err_lo = mean - q16
    err_hi = q84 - mean

    if mean.size == 1:
        return float(mean), float(err_lo), float(err_hi)
    return mean, err_lo, err_hi

import numpy as np

import numpy as np

def _dlnI_dgamma(gamma, Emin, Emax):
    """d(ln I)/dgamma のベクトル対応"""
    a = 2.0 - gamma
    out = np.empty_like(gamma, dtype=float)
    mask_eq2 = np.isclose(a, 0.0)

    # 一般項
    S = np.power(Emax, a) - np.power(Emin, a)
    term = (np.power(Emax, a)*np.log(Emax) - np.power(Emin, a)*np.log(Emin)) / S
    out[~mask_eq2] = 1.0/a[~mask_eq2] - term[~mask_eq2]

    # γ ≈ 2 の場合
    out[mask_eq2] = - (np.log(Emax) + np.log(Emin))
    return out

def pl_lumi_delta(
    f0, f0_err, gamma, gamma_err, d_kpc, d_err,
    E0=1.0, Emin=1.0, Emax=10.0,
    KPC_TO_CM=3.085677581e21, TEV_TO_ERG=1.602176634
):
    # ---- 配列化 & ブロードキャスト ----
    f0, f0_err, g, g_err, d, d_err = np.broadcast_arrays(
        np.asarray(f0, dtype=float),
        np.asarray(f0_err, dtype=float),
        np.asarray(gamma, dtype=float),
        np.asarray(gamma_err, dtype=float),
        np.asarray(d_kpc, dtype=float),
        np.asarray(d_err, dtype=float)
    )

    # 積分 I(g)
    mask_eq2 = np.isclose(g, 2.0)
    I = np.empty_like(g, dtype=float)
    I[mask_eq2] = np.log(Emax / Emin)
    I[~mask_eq2] = (np.power(Emax, 2.0 - g[~mask_eq2]) - np.power(Emin, 2.0 - g[~mask_eq2])) / (2.0 - g[~mask_eq2])

    # 光度
    d_cm = d * KPC_TO_CM
    L = 4.0 * np.pi * (d_cm**2) * f0 * (E0**g) * I * TEV_TO_ERG

    # 偏微分（ln L）
    dlnL_df0 = 1.0 / f0
    dlnL_dd  = 2.0 / d
    dlnL_dg  = np.log(E0) + _dlnI_dgamma(g, Emin, Emax)

    # 分散
    var_lnL = (dlnL_df0 * f0_err)**2 + (dlnL_dd * d_err)**2 + (dlnL_dg * g_err)**2
    sig_L = L * np.sqrt(var_lnL)

    return L, sig_L


# ==============================================================
# 1. 通常ビーミング累積カーブ（未同定源の誤差も考慮）
# ==============================================================

def beaming_curve_cumulative(logL, n_unid_region, logL_nu = None, compress=False, return_error=True):
    """
    通常ビーミング累積カーブ + Poisson誤差（未同定源の不確実性も含む）
    """
    v = np.asarray(logL, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([]), np.array([]), np.nan, 0

    # 高→低に並べ替え
    v_sorted = np.sort(v)[::-1]
    nbins = 10

    # --- ビン化 ---
    if nbins is not None and nbins > 1:
        edges = np.linspace(v_sorted.min(), v_sorted.max(), nbins + 1)
        centers = 0.5 * (edges[1:] + edges[:-1])
        counts = np.array([(v_sorted >= edge).sum() for edge in edges[:-1]])
        x = centers  # 高→低方向
        k = counts
    else:
        if compress:
            uniq, counts = np.unique(v_sorted, return_counts=True)
            x = uniq
            k = np.cumsum(counts)
        else:
            x = v_sorted
            k = np.arange(1, v_sorted.size + 1)

    denom = k + n_unid_region
    y = k / denom
    y_final = y[-1]
    n_det = int(k[-1])

    if not return_error:
        return x, y, y_final, n_det

    # --- Poisson誤差（検出数と未同定数の両方を考慮）---
    sigma_y = np.sqrt(k * n_unid_region) / (denom ** 1.5)

    y_low = np.clip(y - sigma_y, 0, 1)
    y_high = np.clip(y + sigma_y, 0, 1)

    return x, y, y_final, n_det, sigma_y, y_low, y_high


# ==============================================================
# 2. 年齢依存ビーミング累積カーブ（未同定源の誤差も考慮）
# ==============================================================

def beaming_curve_cumulative_age(log_age, n_unid_region, mode="binned10",
                                 return_error=True, prepend_first=True):
    """
    年齢依存ビーミング累積カーブ + Poisson誤差（未同定源の不確実性も含む）
    """
    v = np.sort(np.asarray(log_age, float))  # 昇順
    if v.size == 0:
        return np.array([]), np.array([]), np.nan

    if mode == "ecdf":
        x = v
        k = np.arange(1, v.size + 1)
    else:  # "binned10"
        edges = np.linspace(v[0], v[-1], 11)
        k = np.searchsorted(v, edges[1:], side='right')
        x = 0.5 * (edges[:-1] + edges[1:])
        if prepend_first:
            x = np.concatenate(([v[0]], x))
            k = np.concatenate(([1], k))
            k = np.maximum.accumulate(k)  # 単調増加性の保証

    denom = k + n_unid_region
    y = k / denom
    y_final = y[-1]

    if not return_error:
        return x, y, y_final

    # --- Poisson誤差（検出数と未同定数の両方を考慮）---
    sigma_y = np.sqrt(k * n_unid_region) / (denom ** 1.5)
    y_low = np.clip(y - sigma_y, 0, 1)
    y_high = np.clip(y + sigma_y, 0, 1)

    return x, y, y_final, sigma_y, y_low, y_high

# 互換用ラッパ（binsは無視して累積版を呼ぶ）
def beaming_curve_with_final(logL, bins, n_unid_region, compress=False):
    return beaming_curve_cumulative(logL, n_unid_region, compress=compress)

def draw_box(ax, l1, l2, b1, b2, **kwargs):
    """単純な矩形をAitoff上に描く（lはすでに[-180,180]内の値）"""
    l_vals = np.linspace(l1, l2, 200)
    b_top = np.full_like(l_vals, b2)
    b_bot = np.full_like(l_vals, b1)
    # 上辺
    ax.plot(np.radians(l_vals), np.radians(b_top), **kwargs)
    # 下辺
    ax.plot(np.radians(l_vals), np.radians(b_bot), **kwargs)
    # 左右辺
    ax.plot(np.radians([l1, l1]), np.radians([b1, b2]), **kwargs)
    ax.plot(np.radians([l2, l2]), np.radians([b1, b2]), **kwargs)


def add_lb_box_aitoff(ax, l_min, l_max, b_min, b_max, **kwargs):
    """
    Aitoff投影で銀経銀緯範囲ボックスを描く。
    l_min > l_max の場合（0°をまたぐ）は自動で分割して両側に描く。
    """
    # 銀経を[-180,180]に変換
    l_min_wrap = ((l_min + 180) % 360) - 180
    l_max_wrap = ((l_max + 180) % 360) - 180

    # 色指定
    if "edgecolor" in kwargs:
        kwargs["color"] = kwargs.pop("edgecolor")
    kwargs.pop("facecolor", None)

    # 通常範囲
    if l_min_wrap < l_max_wrap:
        draw_box(ax, l_min_wrap, l_max_wrap, b_min, b_max, **kwargs)
    else:
        # 0°（±180°）をまたぐ → 分割して両側に描く
        draw_box(ax, l_min_wrap, 180, b_min, b_max, **kwargs)
        draw_box(ax, -180, l_max_wrap, b_min, b_max, **kwargs)
