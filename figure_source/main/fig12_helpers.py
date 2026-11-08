from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.text import Text
import paper_style as ps
from chemical_asset_style import draw_asset
ROOT = Path(__file__).resolve().parents[2]
R7 = ROOT
BASE = ROOT/'rendered'
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def save(fig,number,version,data,geometry,script):
    target=BASE/('figure_%d'%number)/version
    target.mkdir(parents=True,exist_ok=True)
    stem='figure_%d'%number
    if (target/(stem+'.png')).exists(): raise FileExistsError('Choose a new output version')
    for text in fig.findobj(Text):
        chemical=bool(getattr(getattr(text,'axes',None),'_chemical_asset_audit',None))
        if not chemical and text.get_fontsize()<7.5: text.set_fontsize(7.5)
    fig.canvas.draw()
    for ext in ['pdf','svg','png']: fig.savefig(str(target/(stem+'.'+ext)),dpi=600,facecolor='white')
    write(target/(stem+'_plot_data.json'),data)
    write(target/(stem+'_geometry.json'),geometry)
    meta={'PNG_pixel_size':[round(float(v)*600) for v in fig.get_size_inches()]}
    plt.close(fig)
    return target,meta
