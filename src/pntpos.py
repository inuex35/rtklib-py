"""
pntpos.py : module for standalone positioning

Copyright (c) 2021 Rui Hirokawa (from CSSRLIB)
Copyright (c) 2022 Tim Everett
"""
import numpy as np
from numpy.linalg import norm, lstsq
from rtkcmn import rCST, ecef2pos, geodist, satazel, ionmodel, tropmodel, \
     Sol, tropmapf, uGNSS, trace, timeadd
import rtkcmn as gn
from ephemeris import seleph, satposs
from rinex import rcvstds

NX =        5           # num of estimated parameters, pos + clock
MAXITR =    10          #  max number of iteration or point pos
ERR_ION =   5.0         #  ionospheric delay Std (m)
ERR_TROP =  3.0         #  tropspheric delay Std (m)
ERR_SAAS =  0.3         #  Saastamoinen model error Std (m)
ERR_BRDCI = 0.5         #  broadcast ionosphere model error factor
ERR_CBIAS = 0.3         #  code bias error Std (m)
REL_HUMI =  0.7         #  relative humidity for Saastamoinen model
MIN_EL = np.deg2rad(5)  #  min elevation for measurement

def varerr(nav, el, rcvstd):
    """ variation of measurement """
    s_el = np.sin(el)
    if s_el <= 0.0:
        return 0.0
    a = 0.003 #nav.err[1]
    b = 0.003 #nav.err[2]
    var = a**2+(b/s_el)**2
    # TODO: add adjustment for constellation, not needed for GPS+GAL only
    # add scaled stdevs from receiver
    #if nav.err[3] > 0:
    #    var += (nav.err[3] * rcvstd)**2
    return var

def prange(nav, obs, i):
    eph = seleph(nav, obs.t, obs.sat[i])
    if obs.P[i, 0] == 0:
        return 0
    P = obs.P[i, 0] - eph.tgd * rCST.CLIGHT
    return P

def rescode(iter, obs, nav, rs, dts, svh, x):
    """ calculate code residuals """
    ns = len(obs.sat)  # measurements
    trace(3, 'rescode : n=%d\n' % ns)
    v = np.zeros(ns + NX - 3)
    H = np.zeros((ns + NX - 3, NX))
    mask = np.zeros(NX - 3) # clk states 
    azv = np.zeros(ns)
    elv = np.zeros(ns)
    var = np.zeros(ns + NX - 3)
    
    rr = x[0:3].copy()
    dtr = x[3]
    pos = ecef2pos(rr)
    rcvstds(nav, obs) # decode stdevs from receiver
    
    nv = 0
    for i in range(ns):
        sys = nav.sysprn[obs.sat[i]][0]
        if norm(rs[i,:]) < rCST.RE_WGS84 or svh[i] > 0:
            continue
        # TODO: excluded satellites
        #if satexclude(obs.sat[i], var):
        #    continue
        # geometric distance and elevation mask
        r, e = geodist(rs[i], rr)
        if r < 0:
            continue
        az, el = satazel(pos, e)
        if el < nav.elmin:
            continue
        if iter > 0:
            # TODO: test SNR mask
            # ionospheric correction
            dion = ionmodel(obs.t, pos, az, el, nav.ion)
            # tropospheric correction
            trop_hs, trop_wet, _ = tropmodel(obs.t, pos, el, REL_HUMI)
            mapfh, mapfw = tropmapf(obs.t, pos, el)
            dtrp = mapfh * trop_hs + mapfw * trop_wet
        else:
            dion = dtrp = 0
        # psendorange with code bias correction
        P = prange(nav, obs, i)
        if P == 0:
            continue
        # pseudorange residual
        v[nv] = P - (r + dtr - rCST.CLIGHT * dts[i] + dion + dtrp)
        # design matrix 
        H[nv, 0:3] = -e
        H[nv, 3] = 1
        # time system offset and receiver bias correction
        if sys == uGNSS.GAL:
            v[nv] -= x[4]
            H[nv, 4] = 1.0
            mask[1] = 1
        else:
            mask[0] = 1
            
        azv[nv] = az
        elv[nv] = el
        var[nv] = varerr(nav, el, nav.rcvstd[obs.sat[i]-1,0])
        nv += 1

    # constraint to avoid rank-deficient
    for i in range(NX - 3):
        if mask[i] == 0:
            v[nv] = 0.0
            H[nv, i+3] = 1
            var[nv] = 0.01
            nv += 1
    v = v[0:nv]
    H = H[0:nv, :]
    azv = azv[0:nv]
    elv = elv[0:nv]
    var = var[0:nv]
    return v, H, azv, elv, var


def estpos(obs, nav, rs, dts, svh):
    """ estimate position and clock errors with standard precision """
    x = np.zeros(NX)
    x[0:3] = nav.x[0:3]
    sol = Sol()
    for iter in range(MAXITR):
        v, H, az, el, var = rescode(iter, obs, nav, rs[:,0:3], dts, svh, x)
        nv = len(v)
        if nv < NX:
            trace(3, 'estpos: lack of valid sats nsat=%d nv=%d\n' % 
                  (len(obs.sat), nv))
            return sol
        # weight by variance (lsq uses sqrt of weight 
        sig = np.sqrt(var)
        v /= sig
        for j in range(nv):
            H[j,:] /= sig[j]
        # least square estimation
        dx = lstsq(H, v, rcond=None)[0]
        x += dx
        if norm(dx) < 1e-4:
            break
    else: # exceeded max iterations
        sol.stat = gn.SOLQ_NONE
        trace(3, 'estpos: solution did not converge\n')
    sol.stat = gn.SOLQ_SINGLE
    sol.t = timeadd(obs.t, -x[3] / rCST.CLIGHT )
    sol.dtr = x[3:5] / rCST.CLIGHT
    sol.rr[0:3] = x[0:3]
    sol.rr[3:6] = 0
    return sol

def pntpos(obs, nav):
    """ single-point positioning ----------------------------------------------------
    * compute receiver position, velocity, clock bias by single-point positioning
    * with pseudorange and doppler observables
    * args   : obs      I   observation data
    *          nav      I   navigation data
    * return : sol      O   """
    rs, _, dts, svh = satposs(obs, nav)
    sol = estpos(obs, nav, rs, dts, svh)
    return sol
    

