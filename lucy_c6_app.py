"""Lucy C6 well design – Streamlit app.

Run with:   streamlit run lucy_c6_app.py
Needs:      streamlit only (numpy, pandas and altair come with it)
"""
import math, json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title='Lucy C6 Well Design', layout='wide')

RIGS = {'Ensco 120': 115000, 'Ensco 70': 98000, 'Transocean Monarch': 105000,
        'Transocean GSF Galaxy II': 124000, 'Rowan Gorilla V': 127000}


# ------------------------------------------------------------------ bit design engine
CD = 0.95          # nozzle discharge coefficient
HSI_FRAC = 0.65    # theoretical optimum (max bit HHP); design uses a lower per-section fraction to leave room for
                   # parasitic losses that grow with depth, underreamer and MPD choke

BITS = [
 dict(sec='26"', hole=26.0, frm=500, to=2500, bit='Milled-tooth roller cone', iadc='115', blades='3 cones', cutters='Steel teeth, hardfaced',
      lith='Soft shale, unconsolidated sand', why='Soft, sticky top hole; aggressive long teeth, low cost; balling resistance from open jets',
      wob='20–40', rpm='80–120', q=1100, mw=9.2, spp=4000, frac=0.55, nn=4, dp=5.5, ann=26.0),
 dict(sec='17-1/2"', hole=17.5, frm=2500, to=5300, bit='PDC (steel body)', iadc='M323', blades='6 blades', cutters='19 mm, single row',
      lith='Reactive shale, sand stringers', why='Shear cutting in shale is fastest; one-run section on motor; OBM prevents balling',
      wob='10–25', rpm='120–180 (motor)', q=1000, mw=10.2, spp=5000, frac=0.50, nn=6, dp=5.5, ann=17.5),
 dict(sec='12-1/4" x 14-1/2"', hole=12.25, frm=5300, to=5900, bit='TCI roller cone + hydraulic underreamer', iadc='517', blades='3 cones', cutters='Chisel inserts, gauge protected',
      lith='Fractured / vugular limestone', why='Impact-tolerant inserts survive fractured carbonate; low torque under MPD and losses',
      wob='30–45', rpm='80–110', q=800, mw=10.6, spp=5000, frac=0.40, nn=3, dp=5.5, ann=14.5),
 dict(sec='10-5/8" x 12-1/4"', hole=10.625, frm=5900, to=11800, bit='PDC (matrix body) + underreamer', iadc='M323', blades='7 blades', cutters='16 mm, backup row',
      lith='Shale, mobile salt', why='Matrix body for erosion; backup cutters for durability; 2 runs (fresh bit before salt); stable on RSS',
      wob='10–25', rpm='140–180 (RSS)', q=750, mw=12.8, spp=5000, frac=0.35, nn=7, dp=5.5, ann=12.25),
 dict(sec='8-1/2"', hole=8.5, frm=11800, to=12500, bit='PDC (matrix body)', iadc='M433', blades='6 blades', cutters='13 mm, impact-resistant',
      lith='Limestone, target sandstone', why='Smaller cutters and more blades for harder, interbedded rock; good dull grade for coring/logging',
      wob='8–20', rpm='120–160 (RSS)', q=500, mw=13.3, spp=5000, frac=0.35, nn=6, dp=5.0, ann=8.5),
]


def tfa(sizes):
    return sum(math.pi / 4 * (s / 32) ** 2 for s in sizes)


def best_nozzles(n, a_req):
    best = None
    for a in range(8, 32):
        for k in range(n + 1):
            sizes = [a] * k + [a + 1] * (n - k)
            err = abs(tfa(sizes) - a_req)
            if best is None or err < best[0]:
                best = (err, sizes)
    return best[1]


def hydraulics(b):
    q, mw = b['q'], b['mw']
    dp_target = b['frac'] * b['spp']
    a_req = math.sqrt(mw * q ** 2 / (12031 * CD ** 2 * dp_target))
    sizes = best_nozzles(b['nn'], a_req)
    a = tfa(sizes)
    dpb = mw * q ** 2 / (12031 * CD ** 2 * a ** 2)
    vn = 0.3208 * q / a
    hhp = dpb * q / 1714
    hsi = hhp / (math.pi / 4 * b['hole'] ** 2)
    imp = 0.01823 * CD * q * math.sqrt(mw * dpb)
    av = 24.51 * q / (b['ann'] ** 2 - b['dp'] ** 2)
    counts = {}
    for s in sizes: counts[s] = counts.get(s, 0) + 1
    noz = ' + '.join(f'{c}×{s}/32"' for s, c in sorted(counts.items()))
    return dict(dp_target=round(dp_target), tfa_req=round(a_req, 3), nozzles=noz, tfa=round(a, 3), dp_bit=round(dpb),
                pct_spp=round(dpb / b['spp'] * 100, 1), vn=round(vn), hhp=round(hhp), hsi=round(hsi, 2), impact=round(imp), av=round(av))


def economics(b, rop_ftday, trip_hr, spread_per_day, bit_rate):
    footage = b['to'] - b['frm']
    drill_hr = footage / rop_ftday * 24
    bit_cost = footage * bit_rate
    cpf = (bit_cost + spread_per_day / 24 * (drill_hr + trip_hr)) / footage
    return dict(footage=footage, drill_hr=round(drill_hr, 1), trip_hr=round(trip_hr, 1), bit_cost=round(bit_cost), cpf=round(cpf))


def drill_hours(a, b):
    h = 0; d = a
    while d < b:
        rop = 900 if d < 6000 else 750 if d < 9000 else 540 if d < 12200 else 380
        nxt = min(b, 6000 if d < 6000 else 9000 if d < 9000 else 12200 if d < 12200 else b)
        h += (nxt - d) / rop * 24; d = nxt
    return h


def design(spread_per_day, stand_ft=90, trip_spd=12):
    out = []
    for b in BITS:
        h = hydraulics(b)
        footage = b['to'] - b['frm']
        drill_hr = drill_hours(b['frm'], b['to'])
        n_runs = 2 if b['sec'].startswith('10-5/8') else 1
        trip_hr = sum(2 * d / stand_ft / trip_spd for d in ([b['frm']] + ([9000] if n_runs == 2 else [])))
        rate = 28 if b['hole'] >= 17.5 else 17
        bit_cost = footage * rate
        cpf = (bit_cost + spread_per_day / 24 * (drill_hr + trip_hr)) / footage
        out.append({**b, **h, 'runs': n_runs, 'footage': footage, 'drill_hr': round(drill_hr, 1), 'trip_hr': round(trip_hr, 1),
                    'bit_cost': round(bit_cost), 'cpf': round(cpf)})
    return out



@st.cache_data
def compute(RIG_NAME, RIG_RATE, NPT, PKF, BACKUP, MW_RUN, CONT):
    G = 0.052
    # ---------------- Geology (Appendix 1) ----------------
    geo = [
     (500,'Shale',8.85,10.80),(1000,'Shale',8.85,11.15),(1200,'Sand',8.85,10.85),(1300,'Shale',8.85,11.69),
     (2000,'Shale',8.85,11.83),(2500,'Shale',8.85,12.22),(3000,'Shale',8.85,12.47),(3700,'Sand',8.85,12.15),
     (3880,'Shale',8.85,13.01),(5000,'Shale',9.28,13.33),(5500,'Limestone',10.05,13.48),(5700,'Shale',10.09,13.93),
     (6000,'Shale',10.40,14.45),(6500,'Shale',10.90,14.79),(7000,'Shale',11.22,15.33),(7500,'Shale',11.51,15.56),
     (8000,'Shale',11.59,15.70),(8500,'Shale',11.61,15.54),(9000,'Shale',11.64,15.73),(9600,'Salt',11.66,16.71),
     (9800,'Salt',11.66,16.71),(10050,'Shale',11.66,15.92),(10500,'Shale',11.69,15.93),(10750,'Shale',11.71,16.13),
     (11000,'Shale',11.74,16.14),(11250,'Shale',11.82,16.28),(11500,'Shale',11.85,16.30),(11750,'Shale',12.34,16.50),
     (12000,'Limestone',12.68,16.00),(12200,'Sand',12.76,15.75),(12500,'Shale',13.01,16.89)]
    D = np.array([g[0] for g in geo]); PP = np.array([g[2] for g in geo]); FG = np.array([g[3] for g in geo])
    lot = [(5000,9.0,13.1),(5150,9.2,13.2),(9400,11.7,16.1),(9900,11.8,15.8),(11900,12.7,16.0)]
    pp = lambda d: float(np.interp(d, D, PP))
    def fg(d):  # design FG = min(appendix, offset LOT where close)
        f = float(np.interp(d, D, FG))
        for ld,_,lv in lot:
            if abs(ld-d) <= 250: f = min(f, lv)
        return f

    # ---------------- Casing scheme ----------------
    # name, OD, wt, grade, ID, drift(used), hole, shoe, top, mud weight in next/this section
    strings = [
     dict(name='30" Conductor',od=30,wt=310,grade='X-52',id=28.0,hole='36"',shoe=500,top=0,mw=None),
     dict(name='20" Surface',od=20,wt=133,grade='K-55 BTC',id=18.730,drift=18.542,hole='26"',shoe=2500,top=0,mw=9.2),
     dict(name='13-3/8" Intermediate',od=13.375,wt=72,grade='L-80 BTC',id=12.347,drift=12.25,hole='17-1/2"',shoe=5300,top=0,mw=10.2),
     dict(name='11-3/4" Drilling liner',od=11.75,wt=60,grade='L-80 premium',id=10.772,drift=10.625,hole='12-1/4" x 14-1/2"',shoe=5900,top=5000,mw=10.6),
     dict(name='9-5/8" x 9-7/8" Production',od=9.625,wt=53.5,grade='P-110 / Q-125',id=8.535,drift=8.5,hole='10-5/8" x 12-1/4"',shoe=11800,top=0,mw=12.8),
     dict(name='7" Production liner',od=7,wt=32,grade='P-110 premium',id=6.094,drift=6.0,hole='8-1/2"',shoe=12500,top=11500,mw=13.3),
    ]
    mudprog = [  # section, interval, type, MW
     ('26"',500,2500,'KCl/PHPA WBM',9.0,9.2),
     ('17-1/2"',2500,5300,'OBM (low-tox, CaCl2 water phase)',10.0,10.2),
     ('12-1/4" x 14-1/2"',5300,5900,'OBM + LCM, MPD (PMCD contingency)',10.4,10.6),
     ('10-5/8" x 12-1/4"',5900,11800,'OBM (salt-inert)',10.8,12.8),
     ('8-1/2"',11800,12500,'OBM reservoir drill-in (sized CaCO3)',13.2,13.3),
    ]

    # ---------------- Kick tolerance ----------------
    def kick_tol(shoe, td, mw, hole_d, bha_od=8.0, ki=0.5, gas=0.1):
        f = fg(shoe); maasp = (f-mw)*G*shoe
        pk = pp(td)+ki; sidpp = max(0,(pk-mw)*G*td)
        H = (maasp - sidpp)/(mw*G - gas)
        cap = (hole_d**2 - bha_od**2)/1029.4
        v_shoe = H*cap
        p_shoe = f*G*shoe; p_bot = pk*G*td
        v_bot = v_shoe*p_shoe/p_bot
        return dict(shoe=shoe, td=td, fg_shoe=round(f,2), mw=mw, maasp=round(maasp), sidpp=round(sidpp),
                    H=round(H), kt_bbl=round(v_bot,1))
    kt = [kick_tol(2500,5300,10.2,17.5,8.0), kick_tol(5300,5900,10.6,14.5,8.0),
          kick_tol(5900,11800,12.8,12.25,6.75), kick_tol(11800,12500,13.3,8.5,6.5)]

    # ---------------- API 5C3 ratings ----------------
    def api_collapse(Dod, t, Yp):
        dt = Dod/t; Y = Yp/1000.0
        A = 2.8762+0.10679e-5*Yp+0.21301e-10*Yp**2-0.53132e-16*Yp**3
        B = 0.026233+0.50609e-6*Yp
        C = -465.93+0.030867*Yp-0.10483e-7*Yp**2+0.36989e-13*Yp**3
        F = (46.95e6*(3*B/A/(2+B/A))**3)/(Yp*((3*B/A)/(2+B/A)-B/A)*(1-(3*B/A)/(2+B/A))**2)
        Gc = F*B/A
        dt_yp = (math.sqrt((A-2)**2+8*(B+C/Yp))+(A-2))/(2*(B+C/Yp))
        dt_pt = Yp*(A-F)/(C+Yp*(B-Gc))
        dt_te = (2+B/A)/(3*B/A)
        if dt <= dt_yp: return 2*Yp*((dt-1)/dt**2), 'yield'
        if dt <= dt_pt: return Yp*(A/dt-B)-C, 'plastic'
        if dt <= dt_te: return Yp*(F/dt-Gc), 'transition'
        return 46.95e6/(dt*(dt-1)**2), 'elastic'
    def burst(Dod,t,Yp): return 0.875*2*Yp*t/Dod
    def body(Dod,t,Yp): return Yp*math.pi/4*(Dod**2-(Dod-2*t)**2)/1000
    cands = {
     '9-5/8" 53.5# P-110': (9.625,0.545,110000,53.5),
     '9-5/8" 53.5# Q-125': (9.625,0.545,125000,53.5),
     '9-7/8" 62.8# Q-125': (9.875,0.625,125000,62.8),
    }
    ratings = {}
    for k,(o,t,y,w) in cands.items():
        c,reg = api_collapse(o,t,y)
        ratings[k] = dict(burst=round(burst(o,t,y),-1), collapse=round(c,-1), regime=reg, body=round(body(o,t,y)), wt=w, id=round(o-2*t,3))
    # other strings quick ratings
    others = {'20" 133# K-55':(20,0.635,55000),'13-3/8" 72# L-80':(13.375,0.514,80000),
              '11-3/4" 60# L-80':(11.75,0.489,80000),'7" 32# P-110':(7,0.453,110000)}
    oth = {k:dict(burst=round(burst(*v),-1),collapse=round(api_collapse(*v)[0],-1),body=round(body(*v))) for k,v in others.items()}

    # ---------------- Production casing loads (9-5/8" x 9-7/8", 0-11,800) ----------------
    shoe = 11800; z = np.arange(0, shoe+1, 100)
    Pres = pp(12200)*G*12200          # reservoir pressure at target
    Pres_td = pp(12500)*G*12500
    gasg = 0.1; pkf = PKF; ext_b = BACKUP
    SITP = Pres - gasg*12200
    B1_int = SITP + pkf*G*z                      # tubing leak below tree, on 10.0 ppg packer fluid
    B2_surf = min(Pres_td - gasg*12500, fg(shoe)*G*shoe - gasg*shoe)
    B2_int = B2_surf + gasg*z                    # gas to surface while drilling 8-1/2" section
    ext_burst = ext_b*G*z                         # degraded mud / seawater backup
    net_burst = np.maximum(B1_int, B2_int) - ext_burst
    # Collapse
    mw_run = MW_RUN
    C1_ext = mw_run*G*z                           # full evacuation, external = mud casing run in
    salt_top, salt_base = 9500, 9900              # incl. +/-100 ft uncertainty
    C2_ext = np.where((z>=salt_top)&(z<=salt_base), 1.0*z, mw_run*G*z)   # salt: overburden 1.0 psi/ft
    C_int = gasg*z*0   # full evacuation (conservative)
    net_collapse = np.maximum(C1_ext, C2_ext) - C_int
    DFb, DFc, DFt = 1.10, 1.10, 1.60
    # Selected tapered string
    taper = 9000  # 9-5/8" 53.5# P-110 0-9,000 ; 9-7/8" 62.8# Q-125 9,000-11,800 (covers salt +/- uncertainty)
    r_top = ratings['9-5/8" 53.5# P-110']; r_bot = ratings['9-7/8" 62.8# Q-125']
    burst_rating = np.where(z<=taper, r_top['burst'], r_bot['burst'])
    coll_rating = np.where(z<=taper, r_top['collapse'], r_bot['collapse'])
    sf_b = burst_rating/np.maximum(net_burst,1); sf_c = coll_rating/np.maximum(net_collapse,1)
    # Tension
    BF = 1 - mw_run/65.5
    w_air = 53.5*taper + 62.8*(shoe-taper)
    w_buoy = w_air*BF
    overpull = 100000
    t_top = w_buoy + overpull
    t_shock = w_buoy + 3200*53.5      # shock load approx 3200*W
    sf_t = r_top['body']*1000/max(t_top, t_shock)
    prod = dict(SITP=round(SITP), Pres=round(Pres), B2_surf=round(B2_surf),
                max_net_burst=round(net_burst.max()), burst_surface=round(net_burst[0]), burst_shoe=round(net_burst[-1]),
                max_net_collapse=round(net_collapse.max()), collapse_shoe=round(net_collapse[-1]),
                collapse_salt=round(net_collapse[z==9900][0]),
                min_sf_burst=round(float(sf_b.min()),2), min_sf_collapse=round(float(sf_c.min()),2),
                w_air=round(w_air), BF=round(BF,3), w_buoy=round(w_buoy), t_top=round(t_top), t_shock=round(t_shock),
                sf_tension=round(sf_t,2), wh_rating='10,000 psi (max SITP %d psi)'%round(SITP))
    # collapse SF if salt load used 53.5# P-110 (justify taper)
    sf_salt_p110 = r_top['collapse']/net_collapse[z==9900][0]
    prod['sf_salt_if_p110'] = round(float(sf_salt_p110),2)
    prod['sf_shoe_coll_if_p110'] = round(r_top['collapse']/net_collapse[-1],2)
    loads = dict(z=z.tolist(), net_burst=net_burst.round().tolist(), net_collapse=net_collapse.round().tolist(),
                 burst_rating=burst_rating.tolist(), coll_rating=coll_rating.tolist())

    # ---------------- Cement ----------------
    cap = lambda a,b: (a**2-b**2)/1029.4
    cem = []
    def add(job, slurry, ppg, top, base, hole_d, od, excess, inside=None):
        if inside: c = cap(inside, od); ex = 0
        else: c = cap(hole_d, od); ex = excess
        v = c*(base-top)*(1+ex)
        cem.append(dict(job=job, slurry=slurry, ppg=ppg, top=top, base=base, bbl=round(v)))
    # 20": to seabed (350 ft RKB). Inside 30" (ID 28) 350-500, open hole 26" 500-2500 (100% excess)
    add('20"','Lead extended',12.5,350,500,None,20,0,inside=28.0)
    add('20"','Lead extended',12.5,500,2000,26,20,1.0)
    add('20"','Tail Class G',15.8,2000,2500,26,20,1.0)
    # 13-3/8": TOC 2,000 (500 ft into 20"), tail across 3,700 sand aquifer
    add('13-3/8"','Lead',12.5,2000,2500,None,13.375,0,inside=18.73)
    add('13-3/8"','Lead',12.5,2500,4300,17.5,13.375,0.5)
    add('13-3/8"','Tail Class G',15.8,4300,5300,17.5,13.375,0.5)
    # 11-3/4" liner: full length to liner top 5,000 ; lightweight across loss zone
    add('11-3/4" liner','Lightweight (microsphere + fibre LCM)',13.5,5000,5300,None,11.75,0,inside=12.347)
    add('11-3/4" liner','Lightweight (microsphere + fibre LCM)',13.5,5300,5600,14.5,11.75,0.5)
    add('11-3/4" liner','Tail Class G',15.8,5600,5900,14.5,11.75,0.5)
    # 9-5/8": TOC 8,600 (≈900 ft above earliest salt top), salt-saturated gas-tight
    add('9-5/8" x 9-7/8"','Salt-tolerant gas-tight',16.0,8600,9000,12.25,9.625,0.3)
    add('9-5/8" x 9-7/8"','Salt-tolerant gas-tight',16.0,9000,11800,12.25,9.875,0.3)
    # 7" liner: to liner top 11,500
    add('7" liner','Gas-tight latex + 35% silica',16.0,11500,11800,None,7,0,inside=8.535)
    add('7" liner','Gas-tight latex + 35% silica',16.0,11800,12500,8.5,7,0.3)
    shoe_tracks = [('20"',80,18.73,15.8),('13-3/8"',80,12.347,15.8),('11-3/4" liner',80,10.772,15.8),('9-5/8" x 9-7/8"',80,8.625,16.0),('7" liner',80,6.094,16.0)]
    for j,L,i,ppg in shoe_tracks:
        cem.append(dict(job=j, slurry='Shoe track', ppg=ppg, top=None, base=None, bbl=round(L*i**2/1029.4)))
    cem_cost = sum(c['bbl']*(42 if c['ppg']<14 else 54) for c in cem)
    # ECD checks during cementing (static EMW at shoe / weak point)
    def emw(cols, d):  # cols list of (top, base, ppg)
        p = sum(G*ppg*(min(b,d)-t) for t,b,ppg in cols if t<d)
        return p/(G*d)
    cem_checks = [
     dict(job='13-3/8"', point='shoe 5,300', emw=round(emw([(0,2000,10.2),(2000,4300,12.5),(4300,5300,15.8)],5300),2), fg=round(fg(5300),2)),
     dict(job='9-5/8"', point='shoe 11,800', emw=round(emw([(0,8600,12.8),(8600,11800,16.0)],11800),2), fg=round(fg(11800),2)),
     dict(job='7"', point='TD 12,500', emw=round(emw([(0,11500,13.3),(11500,12500,16.0)],12500),2), fg=round(fg(12200),2)),
     dict(job='11-3/4"', point='limestone 5,600', emw=round(emw([(0,5000,10.6),(5000,5600,13.5)],5600),2), fg=round(fg(5600),2)),
    ]

    # ---------------- Mud volumes ----------------
    def hv(d, L): return d**2/1029.4*L
    mud = [
     dict(sec='26"', type='WBM', bbl=round((hv(28,500)+hv(26,2000))*3.0+1000)),   # 3x dilution top-hole + surface
     dict(sec='17-1/2"', type='OBM', bbl=round(hv(18.73,2500)+hv(17.5*1.05,2800)*1.5+1500)),
     dict(sec='12-1/4" x 14-1/2"', type='OBM', bbl=round(hv(14.5,600)*1.5+3000)),  # incl. 3,000 bbl losses to limestone
     dict(sec='10-5/8" x 12-1/4"', type='OBM', bbl=round(hv(12.25,5900)*1.5+800)),
     dict(sec='8-1/2"', type='OBM', bbl=round(hv(8.5,700)*1.5+500)),
    ]
    for m in mud: m['cost'] = m['bbl']*(30 if m['type']=='WBM' else 60)

    # ---------------- Time vs depth ----------------
    def rop(d):
        return 900 if d<6000 else 750 if d<9000 else 540 if d<12200 else 380
    def drill_hours(a,b):
        h=0; d=a
        while d<b:
            nxt = min(b, 6000 if d<6000 else 9000 if d<9000 else 12200 if d<12200 else b)
            h += (nxt-d)/rop(d)*24; d=nxt
        return h
    trip = lambda d: d/90/12      # one-way, 12 stands/hr, 90 ft stands
    csg_run = lambda L: L/480     # 12 jts/hr assumption
    ops = []  # (desc, hours, depth_end)
    def op(desc,h,depth): ops.append(dict(op=desc,h=round(h,1),depth=depth))
    d0=500
    # Section 1
    op('26": M/U BHA, RIH',6+trip(500),500)
    op('26": Drill 500-2,500',drill_hours(500,2500),2500)
    op('26": Circulate, POOH',4+trip(2500),2500)
    op('20": Run casing',4+csg_run(2500),2500); op('20": Cement',6,2500); op('20": WOC',30,2500)
    op('Install wellhead, N/U & test BOP',18+20,2500)
    # Section 2
    op('17-1/2": M/U BHA, RIH, drill out, LOT',6+trip(2500)+4+4,2500)
    op('17-1/2": Drill 2,500-5,300',drill_hours(2500,5300),5300)
    op('17-1/2": Circulate, POOH',6+trip(5300),5300)
    op('13-3/8": Run casing',4+csg_run(5300),5300); op('13-3/8": Cement',6,5300); op('13-3/8": WOC',30,5300)
    op('Set seal assy, BOP test',6+20,5300)
    # Section 3
    op('Rig up MPD equipment',24,5300)
    op('12-1/4"x14-1/2": M/U BHA+UR, RIH, drill out, FIT',6+trip(5300)+4+4,5300)
    op('12-1/4"x14-1/2": Drill limestone 5,300-5,900 (MPD)',drill_hours(5300,5900),5900)
    op('12-1/4": Circulate, POOH',6+trip(5900),5900)
    op('11-3/4": Run liner on DP',4+csg_run(900)+trip(5000),5900); op('11-3/4": Cement liner',6,5900); op('11-3/4": WOC, test lap',30+4,5900)
    # Section 4
    op('10-5/8"x12-1/4": M/U BHA+UR, RIH, drill out, FIT',6+trip(5900)+4+4,5900)
    op('10-5/8"x12-1/4": Drill 5,900-9,000',drill_hours(5900,9000),9000)
    op('Bit/BHA trip at 9,000 (pre-salt)',2*trip(9000)+6,9000)
    op('10-5/8"x12-1/4": Drill 9,000-11,800 (salt)',drill_hours(9000,11800),11800)
    op('Circulate, wiper trip through salt',8+12,11800)
    op('POOH',trip(11800),11800)
    op('9-5/8"x9-7/8": Run casing',6+csg_run(11800),11800); op('9-5/8": Cement',8,11800); op('9-5/8": WOC',30,11800)
    op('Set seal assy, BOP test',6+20,11800)
    # Section 5
    op('8-1/2": M/U BHA, RIH, drill out, FIT',6+trip(11800)+4+4,11800)
    op('8-1/2": Drill 11,800-12,500',drill_hours(11800,12500),12500)
    op('8-1/2": Circulate, POOH',6+trip(12500),12500)
    op('Wireline: full open-hole suite + pressures/samples',72,12500)
    op('Wiper trip',2*trip(12500)+4,12500)
    op('7": Run liner on DP',4+csg_run(1000)+trip(11500),12500); op('7": Cement liner',6,12500); op('7": WOC',30,12500)
    op('Clean-out/scraper trip, test, CBL/USIT',2*trip(12500)+6+18,12500)
    # Completion
    op('Completion: run 5-1/2" x 4-1/2" 13Cr completion w/ TCP',48,12500)
    op('N/D BOP, N/U Xmas tree, test',24,12500)
    op('Perforate underbalanced, clean-up, suspend',36,12500)
    tot_h = sum(o['h'] for o in ops)
    npt = NPT
    cum=0; tvd=[]
    for o in ops:
        cum += o['h']*(1+npt); tvd.append((round(cum/24,2), o['depth']))
    dry_days = tot_h/24; total_days = dry_days*(1+npt)
    # section day split (for LWD/RSS charges)
    def sec_days(prefixes):
        return sum(o['h'] for o in ops if any(o['op'].startswith(p) for p in prefixes))/24*(1+npt)
    days_26 = sec_days(['26"','20"','Install'])
    days_17 = sec_days(['17-1/2"','13-3/8"','Set seal assy, BOP test']) - 0  # approx
    drill_days = {
     '26"': drill_hours(500,2500)/24*(1+npt) + 0.5,
     '17-1/2"': drill_hours(2500,5300)/24*(1+npt) + 0.5,
     '12-1/4"': drill_hours(5300,5900)/24*(1+npt) + 0.5,
     '10-5/8"': (drill_hours(5900,11800)+2*trip(9000)+20)/24*(1+npt) + 0.5,
     '8-1/2"': drill_hours(11800,12500)/24*(1+npt) + 0.5,
    }
    # ---------------- AFE ----------------
    rig = dict(name=RIG_NAME, rate=RIG_RATE)
    mob = 10  # 5 days each way
    spread_days = total_days + mob
    heli_per_day = 22000*3/7
    afe = []
    def a(cat,item,cost,basis): afe.append(dict(cat=cat,item=item,cost=round(cost),basis=basis))
    a('Time','Rig (%s)'%rig['name'], rig['rate']*spread_days, '%.1f d ops + %d d mob/demob @ $%sk/d'%(total_days,mob,rig['rate']//1000))
    a('Time','Support vessels',18000*spread_days,'$18k/d')
    a('Time','Other services & personnel',40000*spread_days,'$40k/d')
    a('Time','Helicopters',heli_per_day*spread_days,'3 flights/week @ $22k')
    lwd_days = sum(v for k,v in drill_days.items() if k!='26"')
    a('Directional/LWD','LWD (17-1/2" to TD)',18000*lwd_days,'%.1f days in hole'%lwd_days)
    rss_days = drill_days['12-1/4"']+drill_days['10-5/8"']+drill_days['8-1/2"']
    a('Directional/LWD','Rotary steerable (12-1/4", 10-5/8", 8-1/2")',15000*rss_days,'%.1f d'%rss_days)
    mot_days = drill_days['26"']+drill_days['17-1/2"']
    a('Directional/LWD','Steerable motor (26", 17-1/2")',5000*mot_days,'%.1f d'%mot_days)
    a('Services','MPD (12-1/4" x 14-1/2" section)',700000,'1 section')
    a('Services','Underreamers',2*12000,'2 sections')
    bits = 28*(2000+2800) + 17*(600+5900+700)
    a('Consumables','Drill bits',bits,'$28/ft 26"/17-1/2"; $17/ft smaller')
    mud_cost = sum(m['cost'] for m in mud)
    a('Consumables','Mud',mud_cost,'%d bbl WBM, %d bbl OBM'%(sum(m['bbl'] for m in mud if m['type']=='WBM'),sum(m['bbl'] for m in mud if m['type']=='OBM')))
    a('Consumables','Cement',cem_cost,'%d bbl slurry'%sum(c['bbl'] for c in cem))
    a('Evaluation','Logging - full suite (8-1/2")',450000,'reservoir')
    a('Evaluation','Logging - minimal suite (cased-hole CBL/USIT)',220000,'9-5/8" & 7"')
    casing_prices = [('20" 133# K-55 BTC',2500,180),('13-3/8" 72# L-80 BTC',5300,115),('11-3/4" 60# L-80 premium',900,110),
                     ('9-5/8" 53.5# P-110 semi-flush',9000,105),('9-7/8" 62.8# Q-125 semi-flush',2800,150),('7" 32# P-110 premium',1000,65)]
    for n,L,p in casing_prices: a('Tangibles','Casing '+n, L*1.05*p, '%d ft +5%% @ $%d/ft (assumed)'%(L,p))
    a('Tangibles','Wellhead & accessories',600000,'brief')
    a('Tangibles','Packers / plugs',100000,'brief')
    a('Tangibles','Liner hangers (11-3/4", 7") - assumed',2*150000,'assumption')
    a('Tangibles','Completion tubing & SCSSV - assumed',12000*60+150000,'5-1/2" 13Cr @ $60/ft + SCSSV (assumption)')
    afe_total = sum(x['cost'] for x in afe)
    cont = CONT*afe_total

    # ---------------- Relief well ----------------
    def rw_solve(K=3000, b1=2.5, b2=4.0, inc_end=60, tvd_t=11000, hd_t=3500):
        R1 = 18000/(math.pi*b1); R2 = 18000/(math.pi*b2)
        ie = math.radians(inc_end)
        best=None
        for a_deg in np.arange(5,59,0.01):
            al = math.radians(a_deg)
            tv = K + R1*math.sin(al) + R2*(math.sin(ie)-math.sin(al))
            hd = R1*(1-math.cos(al)) + R2*(math.cos(al)-math.cos(ie))
            # hold length L from TVD equation
            L = (tvd_t - tv)/math.cos(al)
            hd_tot = hd + L*math.sin(al)
            err = abs(hd_tot-hd_t)
            if L>0 and (best is None or err<best[0]): best=(err,a_deg,L)
        _,a_deg,L = best; al=math.radians(a_deg)
        # build survey
        pts=[]; md=0; tv=0; hd=0
        pts.append((0,0,0,0))
        md=K; tv=K; pts.append((md,0,tv,0))
        for i in np.arange(0,a_deg+1e-9,1.0 if a_deg>1 else a_deg):
            pass
        # generate stations every 100 ft MD
        stations=[]
        def add_seg_build(md0,tv0,hd0,i0,i1,R):
            n = max(2,int(abs(i1-i0)/ (180/(math.pi*R)*100))+2)
            out=[]
            for k in range(n+1):
                ii = i0+(i1-i0)*k/n
                dmd = R*math.radians(ii-i0)
                dtv = R*(math.sin(math.radians(ii))-math.sin(math.radians(i0)))
                dhd = R*(math.cos(math.radians(i0))-math.cos(math.radians(ii)))
                out.append((md0+dmd, ii, tv0+dtv, hd0+dhd))
            return out
        st=[(0,0,0,0),(K,0,K,0)]
        b1s = add_seg_build(K,K,0,0,a_deg,R1); st += b1s[1:]
        md1,_,tv1,hd1 = st[-1]
        st.append((md1+L, a_deg, tv1+L*math.cos(al), hd1+L*math.sin(al)))
        md2,_,tv2,hd2 = st[-1]
        b2s = add_seg_build(md2,tv2,hd2,a_deg,inc_end,R2); st += b2s[1:]
        return dict(KOP=K,b1=b1,b2=b2,tangent_inc=round(a_deg,2),tangent_len=round(L),
                    eob1=dict(md=round(md1),tvd=round(tv1),hd=round(hd1)),
                    sob2=dict(md=round(md2),tvd=round(tv2),hd=round(hd2)),
                    target=dict(md=round(st[-1][0]),tvd=round(st[-1][2]),hd=round(st[-1][3]),inc=round(st[-1][1],1)),
                    stations=[tuple(round(x,1) for x in s) for s in st])
    rw = rw_solve()

    return {k: v for k, v in locals().items() if not callable(v)}


# ------------------------------------------------------------------ sidebar
st.sidebar.header('Inputs')
rig_name = st.sidebar.selectbox('Rig', list(RIGS), index=0)
rig_rate = st.sidebar.number_input('Rig day rate ($/day)', value=RIGS[rig_name], step=1000)
npt = st.sidebar.slider('Expected NPT', 0.0, 0.30, 0.125, 0.005, format='%.3f')
cont = st.sidebar.slider('AFE contingency', 0.0, 0.25, 0.10, 0.01)
pkf = st.sidebar.number_input('Packer fluid (ppg)', value=10.0, step=0.1)
backup = st.sidebar.number_input('Burst backup density (ppg)', value=8.6, step=0.1)
mw_run = st.sidebar.number_input('MW when 9-5/8" run (ppg)', value=12.8, step=0.1)
st.sidebar.caption('Depths in ft TVD RKB. Appendix 1 depths read as RKB.')

R = compute(rig_name, rig_rate, npt, pkf, backup, mw_run, cont)

st.title('Lucy C6 – Well Design (UKCS 22/21-a, slot 6)')
c1, c2, c3, c4 = st.columns(4)
c1.metric('TD', '12,500 ft TVD RKB')
c2.metric('Spud → suspension', f"{R['total_days']:.1f} days", f"+{10} d mob/demob", delta_color='off')
c3.metric('AFE (P50)', f"${R['afe_total']/1e6:.2f}M", f"${(R['afe_total']+R['cont'])/1e6:.2f}M with contingency", delta_color='off')
c4.metric('Min kick tolerance', f"{min(k['kt_bbl'] for k in R['kt']):.0f} bbl")

INK = '#1f2933'
def ydepth(title='Depth (ft TVD RKB)', bot=12800):
    return alt.Y('Depth:Q', title=title, scale=alt.Scale(reverse=True, domain=[0, bot]))

def lines(df, x, xtitle, xdom, colors, bot=12800, height=650):
    return alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X(f'{x}:Q', title=xtitle, scale=alt.Scale(domain=xdom)), y=ydepth(bot=bot),
        color=alt.Color('Series:N', scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=alt.Legend(orient='bottom')),
        strokeDash=alt.StrokeDash('Dash:N', legend=None), order='i:Q', detail='Series:N',
        tooltip=['Series', alt.Tooltip(f'{x}:Q', format=',.2f'), alt.Tooltip('Depth:Q', format=',')]).properties(height=height)

def rows(series, xs, ds, dash='solid'):
    return [dict(Series=series, X=x, Depth=d, i=i, Dash=dash) for i, (x, d) in enumerate(zip(xs, ds))]

tabs = st.tabs(['PP / FG / MW', 'Casing & kick tolerance', 'Production casing', 'Bit design', 'Cement & mud', 'Time & AFE', 'Relief well'])

# ------------------------------------------------------------------ PPFG
with tabs[0]:
    D, PP, FG = R['D'], R['PP'], R['FG']
    mw_steps = [(500, 2500, 9.2), (2500, 5300, 10.2), (5300, 5900, 10.6), (5900, 9000, 12.0), (9000, 11000, 12.3), (11000, 11800, 12.8), (11800, 12500, 13.3)]
    mx, md = [], []
    for k, (a_, b_, w) in enumerate(mw_steps):
        mx += [w, w]; md += [a_, b_]
    df = pd.DataFrame(rows('Pore pressure', PP, D) + rows('Fracture gradient', FG, D) + rows('Mud weight', mx, md))
    ch = lines(df, 'X', 'Equivalent density (ppg)', [8, 17.5], {'Pore pressure': '#2563eb', 'Fracture gradient': '#dc2626', 'Mud weight': '#16a34a'})
    lot = alt.Chart(pd.DataFrame([dict(X=l[2], Depth=l[0]) for l in R['lot']])).mark_point(shape='diamond', filled=True, size=80, color='#9333ea').encode(x='X:Q', y=ydepth())
    shoes = pd.DataFrame([dict(Depth=s['shoe'], Label=f"{s['name']} @ {s['shoe']:,}") for s in R['strings']])
    rule = alt.Chart(shoes).mark_rule(strokeDash=[4, 3], color=INK).encode(y=ydepth())
    lab = alt.Chart(shoes).mark_text(align='left', dx=4, dy=-6, fontSize=10).encode(y=ydepth(), x=alt.datum(8), text='Label')
    fig = ch + lot + rule + lab
    col1, col2 = st.columns([1.2, 1])
    col1.altair_chart(fig, width='stretch')
    col2.subheader('Geology (Appendix 1)')
    col2.dataframe(pd.DataFrame(R['geo'], columns=['Depth (ft)', 'Lithology', 'PP (ppg)', 'FG (ppg)']), hide_index=True, height=600)

# ------------------------------------------------------------------ casing & KT
with tabs[1]:
    st.subheader('Casing scheme')
    st.dataframe(pd.DataFrame([{k: s.get(k) for k in ['name', 'od', 'wt', 'grade', 'hole', 'top', 'shoe', 'mw']} for s in R['strings']]), hide_index=True)
    st.subheader('Kick tolerance')
    st.dataframe(pd.DataFrame(R['kt']).rename(columns={'shoe': 'Shoe (ft)', 'td': 'Section TD (ft)', 'fg_shoe': 'FG shoe (ppg)', 'mw': 'MW (ppg)',
                 'maasp': 'MAASP (psi)', 'sidpp': 'SIDPP (psi)', 'H': 'Influx height (ft)', 'kt_bbl': 'Kick tolerance (bbl)'}), hide_index=True)
    st.subheader('Pipe ratings (API 5C3)')
    st.dataframe(pd.DataFrame(R['ratings']).T, width='stretch')

# ------------------------------------------------------------------ production casing
with tabs[2]:
    p = R['prod']
    m1, m2, m3 = st.columns(3)
    m1.metric('Min SF burst', p['min_sf_burst']); m2.metric('Min SF collapse', p['min_sf_collapse']); m3.metric('SF tension', p['sf_tension'])
    z = np.array(R['z'])
    lc = st.columns(2)
    for col, load, rating, ttl, c in [(lc[0], R['net_burst'], R['burst_rating'], 'Burst', '#dc2626'), (lc[1], R['net_collapse'], R['coll_rating'], 'Collapse', '#2563eb')]:
        load = np.asarray(load, float); rating = np.asarray(rating, float)
        df = pd.DataFrame(rows('Net load', load, z) + rows('Design load ×1.1', load * 1.1, z, 'dash') + rows('Rating', rating, z))
        col.markdown(f'**{ttl}** (salt 9,500–9,900 ft shaded)')
        salt = alt.Chart(pd.DataFrame([{'a': 9500, 'b': 9900}])).mark_rect(color='#f9a8d4', opacity=0.4).encode(y=alt.Y('a:Q', scale=alt.Scale(reverse=True, domain=[0, 11800])), y2='b:Q')
        col.altair_chart(salt + lines(df, 'X', 'Pressure (psi)', [0, 15000], {'Net load': c, 'Design load ×1.1': c, 'Rating': INK}, bot=11800, height=550), width='stretch')
    st.json({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in p.items()})

# ------------------------------------------------------------------ bit design
with tabs[3]:
    st.subheader('Bit selection')
    st.dataframe(pd.DataFrame([{'Section': b['sec'], 'Interval (ft)': f"{b['frm']:,}–{b['to']:,}", 'Bit': b['bit'], 'IADC': b['iadc'],
                                'Structure': b['blades'], 'Cutters': b['cutters'], 'Formation': b['lith'], 'WOB (klb)': b['wob'],
                                'RPM': b['rpm'], 'Why': b['why']} for b in BITS]), hide_index=True)
    st.subheader('Hydraulics – edit the inputs')
    st.caption('Bit ΔP target = fraction × max standpipe pressure. 0.65 is the theoretical max-bit-HHP optimum; '
               'lower fractions leave room for parasitic losses that grow with depth, the underreamer and the MPD choke. '
               'ΔPbit = MW·Q²/(12,031·Cd²·TFA²), Cd = 0.95.')
    ed = st.data_editor(pd.DataFrame([{'Section': b['sec'], 'Flow (gpm)': b['q'], 'MW (ppg)': b['mw'], 'Max SPP (psi)': b['spp'],
                                       'Bit ΔP fraction': b['frac'], 'Nozzles': b['nn']} for b in BITS]),
                        hide_index=True, disabled=['Section'], key='bit_inputs')
    spread = rig_rate + 18000 + 40000 + 22000 * 3 / 7
    rows_h, rows_e = [], []
    for b, (_, r) in zip(BITS, ed.iterrows()):
        bb = dict(b, q=float(r['Flow (gpm)']), mw=float(r['MW (ppg)']), spp=float(r['Max SPP (psi)']),
                  frac=float(r['Bit ΔP fraction']), nn=int(r['Nozzles']))
        h = hydraulics(bb)
        rows_h.append({'Section': b['sec'], 'Required TFA (in²)': h['tfa_req'], 'Nozzles': h['nozzles'], 'TFA (in²)': h['tfa'],
                       'Bit ΔP (psi)': h['dp_bit'], '% of SPP': h['pct_spp'], 'Jet velocity (ft/s)': h['vn'], 'Bit HHP': h['hhp'],
                       'HSI (hp/in²)': h['hsi'], 'Impact force (lbf)': h['impact'], 'Annular vel. vs DP (ft/min)': h['av']})
        footage = b['to'] - b['frm']; dh = drill_hours(b['frm'], b['to'])
        runs = 2 if b['sec'].startswith('10-5/8') else 1
        th = sum(2 * d / 90 / 12 for d in ([b['frm']] + ([9000] if runs == 2 else [])))
        bc = footage * (28 if b['hole'] >= 17.5 else 17)
        rows_e.append({'Section': b['sec'], 'Bit runs': runs, 'Footage (ft)': footage, 'Drilling (hr)': round(dh, 1), 'Trips (hr)': round(th, 1),
                       'Bit cost ($)': bc, 'Cost per foot ($/ft)': round((bc + spread / 24 * (dh + th)) / footage)})
    st.dataframe(pd.DataFrame(rows_h), hide_index=True)
    st.subheader('Bit economics')
    st.caption(f'Cost per foot = (bit cost + spread rate × (drilling + trip hours)) / footage. Spread rate ${spread:,.0f}/day '
               f'(rig + vessels + services + helicopters). Bit cost from the brief: $28/ft ≥ 17-1/2", $17/ft smaller.')
    st.dataframe(pd.DataFrame(rows_e), hide_index=True)

# ------------------------------------------------------------------ cement & mud
with tabs[4]:
    st.subheader('Cement'); cem = pd.DataFrame(R['cem']); st.dataframe(cem, hide_index=True)
    st.write(f"Total slurry {cem['bbl'].sum():,.0f} bbl · cost ${R['cem_cost']:,.0f}")
    st.dataframe(pd.DataFrame(R['cem_checks']).rename(columns={'emw': 'EMW (ppg)', 'fg': 'FG (ppg)'}), hide_index=True)
    st.subheader('Mud'); st.dataframe(pd.DataFrame(R['mud']), hide_index=True)

# ------------------------------------------------------------------ time & AFE
with tabs[5]:
    t = [0] + [x[0] for x in R['tvd']]; d = [500] + [x[1] for x in R['tvd']]
    df = pd.DataFrame({'Days': t, 'Depth': d, 'i': range(len(t))})
    st.altair_chart(alt.Chart(df).mark_line(strokeWidth=2.5, color=INK).encode(
        x=alt.X('Days:Q', title=f'Days from spud (incl. {npt:.1%} NPT)'), y=ydepth('Depth (ft RKB)', 13000), order='i:Q',
        tooltip=[alt.Tooltip('Days:Q', format='.1f'), alt.Tooltip('Depth:Q', format=',')]).properties(height=420), width='stretch')
    with st.expander('Operations sequence'):
        st.dataframe(pd.DataFrame(R['ops']), hide_index=True)
    st.subheader('AFE')
    afe = pd.DataFrame(R['afe'])
    st.dataframe(afe, hide_index=True, column_config={'cost': st.column_config.NumberColumn('Cost ($)', format='$%d')})
    st.write(f"**Total ${R['afe_total']:,.0f}** · contingency ${R['cont']:,.0f} · **with contingency ${R['afe_total'] + R['cont']:,.0f}**")
    st.bar_chart(afe.groupby('cat')['cost'].sum())

# ------------------------------------------------------------------ relief well
with tabs[6]:
    rw = R['rw']; stn = np.array(rw['stations'])
    a, b = st.columns([1.3, 1])
    df = pd.DataFrame(rows('Relief well', stn[:, 3], stn[:, 2]) + rows('Lucy C6', [3500, 3500], [0, 12000]))
    tgt = alt.Chart(pd.DataFrame([dict(X=3500, Depth=11000)])).mark_point(filled=True, size=120, color='#9333ea').encode(x='X:Q', y=ydepth('TVD (ft)', 12000))
    a.altair_chart(lines(df, 'X', 'Vertical section toward C6, az 315° (ft)', [-300, 4500], {'Relief well': '#dc2626', 'Lucy C6': INK}, bot=12000, height=550) + tgt, width='stretch')
    b.dataframe(pd.DataFrame([
        {'Point': 'KOP', 'MD': rw['KOP'], 'TVD': rw['KOP'], 'HD': 0, 'Inc': 0},
        {'Point': 'EOB', **{k.upper(): v for k, v in rw['eob1'].items()}, 'Inc': rw['tangent_inc']},
        {'Point': 'Start build 2', **{k.upper(): v for k, v in rw['sob2'].items()}, 'Inc': rw['tangent_inc']},
        {'Point': 'Interest point', 'MD': rw['target']['md'], 'TVD': rw['target']['tvd'], 'HD': rw['target']['hd'], 'Inc': rw['target']['inc']},
    ]), hide_index=True)
    b.write(f"Build 1: {rw['b1']}°/100 ft · Build 2: {rw['b2']}°/100 ft · tangent {rw['tangent_inc']:.1f}° for {rw['tangent_len']:,} ft")
