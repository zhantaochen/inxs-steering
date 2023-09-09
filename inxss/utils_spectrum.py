import numpy as np

def formfact(Q,H,K,L,a,c):
    s2 = (Q/4/np.pi)**2
    A0 = 0.0163;    a0 = 35.8826;    B0 = 0.3916;    b0 = 13.2233;    C0 = 0.6052;    c0 = 4.3388;    D0 = -0.0133
    j0 = A0 * np.exp(-a0*s2) + B0 * np.exp(-b0*s2) + C0 * np.exp(-c0*s2) + D0
    A4 = -0.3803;    a4 = 10.4033;    B4 = 0.2838;    b4 = 3.3780;    C4 = 0.2108;    c4 = 1.1036;    D4 = 0.0050
    j4 = (A4 * np.exp(-a4*s2) + B4 * np.exp(-b4*s2) + C4 * np.exp(-c4*s2) + D4) * s2
    y = j0 + j4 * 3/2 * (
        H**4 + K**4 + L**4 * (a/c)**4 - 3 * (H**2 * K**2 + H**2 * L**2 * (a/c)**2 + K**2 * L**2 * (a/c)**2)
        ) / (H**2 + K**2 + L**2 * (a/c)**2 + 1e-15) ** 2
    return y

#TODO: HIGHER BZ
def calc_Sqw_from_SpinW_results(Q, Syy, Szz):
    a = 3.89 # lattice vector in Angstroem in square-lattice notation
    c = 12.55 # lattice vector in Angstroem in square-lattice notation
    if Q.shape[1] != 3:
        Q = Q.T # Scattering vector in r.l.u.
    H = Q[:,0]
    K = Q[:,1]
    L = Q[:,2]
    QL = 2 * np.pi * L / c # Out of plane component of the scattering vector
    Q = 2 * np.pi * np.sqrt((H**2 + K**2) / a**2 + L**2 / c**2) # Scattering vector in Angstroem^-1
    
    # h = np.abs(H - np.round(H)) # Reduced reciprocal lattice vectors projected into the first quadrant of the Brillouin zone
    # k = np.abs(K - np.round(K))
    # l = np.abs(L - np.round(L))
    # l = np.zeros(L.shape)

    S = (np.abs(formfact(Q,H,K,L,a,c))**2)[None,:] * (
            (1 + (QL/(Q+1e-15))**2)[None,:] / 2 * Syy + (1 - (QL/(Q+1e-15))**2)[None,:] * Szz
        )
    return S

def calc_Sqw_from_Syy_Szz(Qw, Syy_func, Szz_func):
    a = 3.89 # lattice vector in Angstroem in square-lattice notation
    c = 12.55 # lattice vector in Angstroem in square-lattice notation
    H, K, L, w = Qw[...,0], Qw[...,1], Qw[...,2], Qw[...,3]
    QL = 2 * np.pi * L / c # Out of plane component of the scattering vector
    Q = 2 * np.pi * np.sqrt((H**2 + K**2) / a**2 + L**2 / c**2) # Scattering vector in Angstroem^-1
    
    # h = np.abs(H - np.round(H)) # Reduced reciprocal lattice vectors projected into the first quadrant of the Brillouin zone
    # k = np.abs(K - np.round(K))
    # # l = np.abs(L - np.round(L))
    # l = np.zeros(L.shape)

    _Qw = Qw.clone()
    _Qw[...,:3] = np.abs(_Qw[...,:3] - np.round(_Qw[...,:3]))
    
    # S = Szz_func(_Qw[...,[0,1,3]])
    S = (np.abs(formfact(Q,H,K,L,a,c))**2) * (
            (1 + (QL/(Q+1e-15))**2) / 2 * Syy_func(_Qw[...,[0,1,3]]) + (1 - (QL/(Q+1e-15))**2) * Szz_func(_Qw[...,[0,1,3]])
        )
    # S = (np.abs(formfact(Q,H,K,L,a,c))**2)
    # S = (
    #         (1 + (QL/(Q+1e-15))**2) / 2 + (1 - (QL/(Q+1e-15))**2)
    #     )
    return S
