from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
ROOT=Path(__file__).resolve().parents[2]
K = 39
def examples():
    return {model:json.loads((ROOT/'source_data'/('figure_S_C%d_plot_data.json'%i)).read_text(encoding='utf-8'))['examples']
            for i,model in [(2,'chemberta_mlm'),(3,'chemberta_mtr'),(4,'chemeleon')]}
def v4_group(group,layer):
    with np.load(str(ROOT/'source_data/figure_S_C5_matrices'/('v4_%s_%s.npz'%(group,layer))),allow_pickle=False) as z:
        return {key:z[key] for key in z.files}
def save(fig,name):
    target=ROOT/'rendered/historical';target.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']: fig.savefig(str(target/(name+'.'+ext)),bbox_inches='tight',pad_inches=.02)
NAMES = {'chemberta_mlm': 'ChemBERTa-2 MLM', 'chemberta_mtr': 'ChemBERTa-2 MTR', 'chemeleon': 'CheMeleon', 'molclr_gin': 'MolCLR', 'hu_masking': 'Hu masking', 'hu_edgepred': 'Hu edgepred', 'hu_contextpred': 'Hu contextpred', 'hu_infomax': 'Hu infomax', 'molformer': 'MoLFormer'}
OI = {'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7', 'black': '#000000', 'grey': '#999999'}
MODEL_COLOR = {'chemberta_mlm': OI['blue'], 'chemberta_mtr': OI['sky'], 'chemeleon': OI['vermillion'], 'molclr_gin': OI['green'], 'hu_masking': OI['purple'], 'hu_edgepred': OI['purple'], 'hu_contextpred': OI['purple'], 'hu_infomax': OI['purple'], 'molformer': OI['orange']}

def setup_style():
    import matplotlib as mpl
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 0.9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 300, 'figure.dpi': 150})

def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')

def effect(S0, others):
    """E = median over concepts of log S0 - median_k log S_k (natural log)."""
    return float(np.median(np.log(S0) - np.median(np.log(np.vstack(others)), axis=0)))

def did(nat, rots):
    """Delta = log S_native - median_r log S_rot, per concept."""
    return np.log(nat) - np.median(np.log(np.vstack(rots)), axis=0)

def loo_effects(S_list):
    """Null cloud: each competitor in turn against the other K-1."""
    return np.array([effect(S_list[k], [S_list[j] for j in range(len(S_list)) if j != k]) for k in range(len(S_list))])

def rank_T(S0, others):
    """T for the target and every competitor (mean per-concept rank), exactly as gates_v2.exact_test."""
    allS = np.vstack([S0] + list(others))
    M = allS.shape[0]
    u = np.empty_like(allS)
    for i in range(M):
        less = (allS < allS[i]).sum(axis=0)
        eq = (allS == allS[i]).sum(axis=0) - 1
        u[i] = (less + 0.5 * eq) / (M - 1)
    return u.mean(axis=1)

def check_against(value, reference, what, tol=1e-09):
    if abs(value - reference) > tol:
        raise AssertionError(f'{what}: recomputed {value} != frozen {reference}')

def joint_effects(d, m, cols=None):
    """J1 and J2 effects for model m plus the leave-one-out twin null clouds."""
    c = slice(None) if cols is None else cols
    tn = [x[c] for x in d['twin_nat']]
    td = [did(d['twin_nat'][k][c], d['twin_rot'][k][:, c]) for k in range(K)]
    s0 = d[f'{m}_nat'][c]
    d0 = did(s0, d[f'{m}_rot'][:, c])
    e1 = effect(s0, tn)
    e2 = float(np.median(d0 - np.median(np.vstack(td), axis=0)))
    null1 = loo_effects(tn)
    null2 = np.array([float(np.median(td[k] - np.median(np.vstack([td[j] for j in range(K) if j != k]), axis=0))) for k in range(K)])
    T1 = rank_T(s0, tn)
    T2 = rank_T(np.exp(d0), [np.exp(x) for x in td])
    return dict(E1=e1, E2=e2, null1=null1, null2=null2, T1=T1, T2=T2)
