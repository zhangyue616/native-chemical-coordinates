from pathlib import Path
from fig12_helpers import ROOT,BASE,ps,read,save as save_number
FROZEN=ROOT/'figure_source/assets/figure_4'
def save(fig,version,data,geometry,script):
    return save_number(fig,4,version,data,geometry,script)
