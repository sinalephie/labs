#LIBRERIE:
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import numpy as np
from IPython.display import Audio, HTML, Latex, display, Image
import math
import seaborn as sns
from numpy import sin,cos,tan,arcsin,arctan,arccos
import requests
import base64
import time
from io import BytesIO
from termcolor import colored
from labs.librerie_aggiuntive.uncertainties.core import ufloat, correlated_values
try:
  from google.colab import output
  is_dark = output.eval_js('document.documentElement.matches("[theme=dark]")')
except ModuleNotFoundError:
  pass
from collections.abc import Sequence

#VARIABILI
# Immagini
immagini = {}
immagini['capra'] = 'https://styles.redditmedia.com/t5_2qlyf/styles/communityIcon_w3vaehlvt5i11.jpg'
immagini['mucca'] = 'https://www.fondazionevb.org/media/cache/582_436_inset/uploads/contents/dona-una-mucca-alle-donne-di-mutanu_1679318060.png'
immagini['cane 1'] = 'https://drive.google.com/file/d/13XIuH3mwCGPDwGVANTQIFtnO7lTjkz0h/view?usp=drive_link'
immagini['cane 2'] = 'https://drive.usercontent.google.com/download?id=1e9JfWwc6xPF1V5HE6CAHRpazE4pDyAxE&export=download&authuser=0&confirm=t&uuid=30b67e8b-41c6-454f-a07f-240c1ad6ae7d&at=APZUnTUd_keQ6z3BDJC152EsBL7Y:1707948416981'
immagini['cane 3'] = 'https://drive.google.com/file/d/16lqpCgaOhYu3yJHw-sP2YkgqDr-wCgzE/view?usp=drive_link'
immagini['gatto 1'] = 'https://drive.google.com/file/d/1lIVYcg258xQA1NJWpw9RRBpb0R0cUBgX/view?usp=drive_link'
immagini['gatto 2'] = 'https://drive.google.com/file/d/1edyHMal4R80Wvh8OnO1aNPBbMPjZMoz1/view?usp=drive_link'
immagini['gatto 3'] = 'https://drive.google.com/file/d/1gW2re7MlGPX9IS-y9H6V-eb7pqGbifh3/view?usp=drive_link'
immagini['gatto 4'] = 'https://drive.google.com/file/d/183wJ66og8102hvZFpWf6fPA07DzJIq6h/view?usp=drive_link'
immagini['gatto 5'] = 'https://drive.google.com/file/d/14gHH1GV_AGa833MZh1xFxVaE8dNMEXgC/view?usp=drive_link'
immagini['gatto 6'] = 'https://drive.google.com/file/d/11Z53qovCo2rfDHnTr0wRVwTIOssNLblu/view?usp=drive_link'
immagini['gatto 7'] = 'https://drive.google.com/file/d/1nH6HOuMwW7Vgj5Jl2_gWkkyEsK5WqbQP/view?usp=drive_link'
immagini['gatto 8'] = 'https://drive.google.com/file/d/1vVn8qLd58CibmhHeZE8m1o9V5XhmPICJ/view?usp=drive_link'
immagini['peppa pig'] =  'https://drive.google.com/file/d/1j_YUDemBk1yfvMVCVs8EyzrTh1IuNZdJ/view?usp=drive_link'
immagini['oca'] = 'https://drive.google.com/file/d/1BmNPsgNSJs1rVLp1uk19zrFTE-U0g6BO/view?usp=drive_link'
immagini['random 1'] = 'https://drive.google.com/file/d/1HqYKUZ7jTMPTc53aPCv1nFhk4CvgZiVV/view?usp=drive_link'
immagini['random 2'] = 'https://drive.google.com/file/d/1zKswYGgEfyE7EskCfPXTxbhYhAREKmXY/view?usp=drive_link'
def colorimatplotlib():
  return print('https://i.stack.imgur.com/lFZum.png')

 # Suoni
suoni = {}
suoni['capra'] = 'https://cdn.pixabay.com/download/audio/2023/11/20/audio_6d2ecb8b19.mp3?filename=goat-sound-177346.mp3'
suoni['risposta corretta'] = 'https://cdn.pixabay.com/download/audio/2021/08/04/audio_bb630cc098.mp3?filename=short-success-sound-glockenspiel-treasure-video-game-6346.mp3'
suoni['yumi'] = 'https://drive.usercontent.google.com/download?id=1TYoZtfyXbDXGvBavp9RiizbyN5c3U_-A&export=download&authuser=0&confirm=t&uuid=3a4c1d91-4195-4fae-b9e5-0597cb84ab03&at=APZUnTUABAkfF1JVWil1TEKAXY4Y:1707948945711'
def box(*args, fontsize=10, loc='upper right', boxstyle='round', facecolor='white', alpha=0.8):
    textstr = '\n'.join(args)
    props = dict(boxstyle=boxstyle, facecolor=facecolor, alpha=alpha)
    loc_dict = {
        'upper right': (0.95, 0.95),
        'upper left': (0.05, 0.95),
        'lower right': (0.95, 0.05),
        'lower left': (0.05, 0.05)
    }
    if loc not in loc_dict:
        raise ValueError(f"loc deve essere una di {list(loc_dict.keys())}")
    x, y = loc_dict[loc]
    plt.gca().text(x, y, textstr, transform=plt.gca().transAxes, fontsize=fontsize,
                   verticalalignment='top', horizontalalignment='right' if 'right' in loc else 'left', bbox=props)


def seleziona_colore():
  import ipywidgets as widgets
  from IPython.display import display
  print('seleziona un colore dall riquadro quadrato e poi premi invio per avere il codice del colore in HTML')
  color_picker = widgets.ColorPicker(concise=False,disabled=False)
  display(color_picker)

def complementare(string):
  string=string[1:]
  r=int(string[0:2],16)
  g=int(string[2:4],16)
  b=int(string[4:6],16)
  comp_r=255-r
  comp_g=255-g
  comp_b=255-b
  comp_html_color = f"#{comp_r:02X}{comp_g:02X}{comp_b:02X}"
  return comp_html_color

def riordina(*args,**kwargs):
    import numpy as np
    args=list(args)
    if 'rispetto' not in kwargs:
        kwargs['rispetto']=args[0]
    if len(args)==1:
      return np.array(sorted(args[0]))
    x=kwargs['rispetto']
    copia=list(x)
    copia2=list(x)
    indici=[]
    for c in range(len(x)):
        minimo=min(copia)
        copia.remove(minimo)
        indice=copia2.index(minimo)
        copia2[indice]=minimo*0.012467421
        indici.append(indice)
    import numpy as np
    lista=[]
    for c in range(len(args)):
        args[c]=np.array(args[c])
        lista.append(np.array(args[c][indici]))
    if len(args)==0:
        return np.array(lista[0])
    else:
        return np.array(lista)
def stile(*args,glow_linee=False,glow_punti=False,riempimento=False,colore_bordo='white',colore_sfondo='white',notazione_scientifica=False,colore_assi=None):
  if colore_bordo != 'white':
    plt.gcf().set_facecolor(f'{colore_bordo}')
  if colore_assi:
    ax = plt.gca()  
    ax.spines['top'].set_color(colore_assi)   
    ax.spines['right'].set_color(colore_assi)  
    ax.spines['left'].set_color(colore_assi)   
    ax.spines['bottom'].set_color(colore_assi)  
    ax.tick_params(axis='both', colors=colore_assi)  
  if colore_sfondo != 'white':
    plt.gca().set_facecolor(f'{colore_sfondo}')
    
  from labs.librerie_aggiuntive.cyberpunk import make_lines_glow, add_underglow, make_scatter_glow
  if glow_punti:
    make_scatter_glow()
  if glow_linee:
    make_lines_glow()
  if riempimento:
    add_underglow()
  if len(args)!=0:
    plt.style.use('default')
    for a in args: 
      c=['default','labs/librerie_aggiuntive/SciencePlots/stile.mplstyle','labs/librerie_aggiuntive/SciencePlots/stile 2.mplstyle','labs/librerie_aggiuntive/SciencePlots/science.mplstyle','Solarize_Light2','_classic_test_patch','bmh','classic', 'dark_background','fast','fivethirtyeight','ggplot','grayscale','seaborn-v0_8','seaborn-v0_8-bright','seaborn-v0_8-colorblind','seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette','seaborn-v0_8-darkgrid','seaborn-v0_8-deep','seaborn-v0_8-muted','seaborn-v0_8-notebook','seaborn-v0_8-paper','seaborn-v0_8-pastel', 'seaborn-v0_8-poster','seaborn-v0_8-talk','seaborn-v0_8-ticks','seaborn-v0_8-white','seaborn-v0_8-whitegrid','tableau-colorblind10']
      if isinstance(a,int):
        try:
          plt.style.use(c[a])
        except:
        
          plt.style.use('default')
      else:
        try:
          if a=='assi cartesiani':
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()
            xmin, xmax = xlim
            ymin, ymax = ylim
            plt.ylim(ymin, ymax)
            plt.xlim(xmin, xmax)
            if xmin > 0:
                xmin = -1
            if ymin > 0:
                ymin = -1
            if xmax < 0:
                xmax = 1
            if ymax < 0:
                ymax = 1
            plt.annotate('', xy=(xmax, 0), xytext=(xmin, 0),
                         arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1))
            plt.annotate('', xy=(0, ymax), xytext=(0, ymin),
                         arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1))
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(0, color='black', linewidth=0.5)
            plt.grid(color='gray', linestyle='--', linewidth=0.5)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)

          if a=='cartone':
            import logging
            import warnings
            logging.getLogger('matplotlib').setLevel(logging.ERROR)
            logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
            with warnings.catch_warnings():
              warnings.simplefilter("ignore")
              plt.xkcd(scale=1,length=1,randomness=1.073)
          else:
            ciao=f'labs/librerie_aggiuntive/SciencePlots/{a}.mplstyle'
            plt.style.use(ciao)
        except:
          plt.style.use('default')
  if plt.gca().get_xscale() != 'log' and plt.gca().get_yscale() != 'log':
    plt.ticklabel_format(useMathText=True)
  if notazione_scientifica:
    def formattazione(x, pos):
      if np.isnan(x) or x == 0:
          return "0"
      exponent = int(np.floor(np.log10(abs(x))))
      coeff = x / 10**exponent
      return rf'${coeff:.0f}_{{\times10^{{{exponent}}}}}$'
    import matplotlib.ticker as mticker
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(formattazione))
    plt.gca().tick_params(axis='y', labelsize=13) 
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(formattazione))
    plt.gca().tick_params(axis='x', labelsize=13) 
def linkdiretto(link):
  import os, re, urllib,shutil
  try:
      urllib.request.urlopen("https://www.google.com", timeout=3)
  except:
      raise ImportError('Sei offline, connettiti ad internet per importare dati')
      return
  if 'google' in link and 'file' in link and 'export' not in link:
    file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    file_id = file_id_match.group(1) if file_id_match else None
    link = f"https://drive.google.com/uc?export=download&id={file_id}"
  if 'google' in link and 'docs' in link and 'spreadsheets' in link and 'usercontent' not in link and 'export' not in link:
    file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    file_id = file_id_match.group(1) if file_id_match else None
    link='https://docs.google.com/spreadsheets/d/'+ file_id + '/export?format=xlsx'
  if '1drv' in link:
    if 'files' in link:
      link='https://api.onedrive.com/v1.0/shares/s!'+ link[link.find('!')+1:link.find('?')]+'/root/content'
  if 'sharepoint' in link:
    if "?e=" in link:
        link = link.split("?e=")[0] + "?download=1"
  return link

def barra_progresso(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%;color: 'blue';'
        >
            {value}
        </progress>
    """.format(value=value, max=max,color='blue'))

def guarda(*links,**kwargs):
  if 'latex' not in kwargs:
    kwargs['latex']=False
  if 'size' not in kwargs:
    kwargs['size']=185
  html_code = "<div style=\"display: flex;\">"
  if not links or 'elenco' in links: 
    out = display(barra_progresso(0, len(immagini)+1), display_id=True)
    progresso=0
    dimensione = 50
    altezza_thumbnail = 50  
    html_code1 = """
    <style>
        .immagine-contenitore {
            display: flex;
        }
        .immagine-contenitore figure {
            margin-right: 2px;
            text-align: center;
        }
        .immagine-contenitore figure img {
            height: auto;  /* Impostare l'altezza su auto per consentire il ridimensionamento proporzionale */
            max-height: """ + str(altezza_thumbnail) + """px;  /* Impostare l'altezza massima per limitare la dimensione */
        }
    </style>
    <div class="immagine-contenitore">
    """
    for nome, link in immagini.items():
        link=linkdiretto(link)
        response = requests.get(link)
        image_data = response.content
        base64_encoded_image = base64.b64encode(image_data).decode('utf-8')
        link = f"data:image/jpeg;base64,{base64_encoded_image}"
        progresso+=1
        out.update(barra_progresso(progresso, len(immagini)+1))
        html_code1 += f"""
        <figure>
            <img src='{link}' alt='{nome}'>
            <figcaption>{nome}</figcaption>
        </figure>
        """
    html_code1 += "</div>"
    progresso+=1
    display(HTML(html_code1))
    out.update(barra_progresso(progresso, len(immagini)+1))
    print('')
    return
  for link in links:
    presente=False
    for c, d in immagini.items():
      if link==c:
        link=d
        presente=True
    if isinstance(link, str) and presente==False and len(link)<30:
      raise ValueError(f'{link} non è stata/o ancora aggiunta/o, per visualizzare l\'elenco con tutte le immagini metti in argomento \'elenco\'')
      return
    link=linkdiretto(link)
    response = requests.get(link)
    if kwargs['latex']==True:
      data = BytesIO(response.content)
      img = mpimg.imread(data)
      ratio = (img.shape[1] / img.shape[0])
      plt.figure(figsize=(ratio*kwargs['size']*0.01, kwargs['size']*0.01))
      plt.imshow(img)
      plt.axis('off')
      plt.show()
      return
    else:
      image_data = response.content
      base64_encoded_image = base64.b64encode(image_data).decode('utf-8')
      immagine = f"data:image/jpeg;base64,{base64_encoded_image}"
      html_code += f"\n    <img src='{immagine}' style='margin-right: 10px; height: {kwargs['size']}px;'>"
  html_code += "\n</div>"
  display(HTML(html_code))
  print('')


#FUNZIONI
def importa_old(link): #obsoleta
  import os, re, urllib,shutil
  import pandas as pd
  
  try:
      urllib.request.urlopen("https://www.google.com", timeout=3)
  except:
      raise ImportError('Sei offline, connettiti ad internet per importare i dati')
      return
  if 'google' and 'edit' and not 'docs' in link:
    file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    file_id = file_id_match.group(1) if file_id_match else None
    link = f"https://drive.google.com/uc?export=download&id={file_id}"
  if 'google' and 'docs' in link:
    file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', link)
    file_id = file_id_match.group(1) if file_id_match else None
    link='https://docs.google.com/spreadsheets/d/'+ file_id + '/export?format=xlsx'
  if '1drv' in link:
    if 'files' in link:
      link='https://api.onedrive.com/v1.0/shares/s!'+ link[link.find('!')+1:link.find('?')]+'/root/content'
  if not os.path.exists('Datiis'):
    os.makedirs('Datiis')
  percorso=os.path.join(os.getcwd(), 'Datiis')
  percorso=percorso + '/'
  c='dati.xlsx'
  urllib.request.urlretrieve(link, percorso+c)
  Dataframe=pd.read_excel(f'{percorso+c}')
  shutil.rmtree('Datiis')
  return Dataframe

def importa(link):
  import os, re, urllib,shutil
  import pandas as pd
  link=linkdiretto(link)
  response = requests.get(link)
  data = BytesIO(response.content)
  Dataframe=pd.read_excel(data)
  return Dataframe
  
def guida():
    print('link alla guida funzioni \u2193 \u2193 \u2193')
    print('https://colab.research.google.com/drive/1Lace8ZenxKYWlCYEODxErVbPpABbGp4G?usp=sharing')

def interpola(x,y,riempi=False,riempi_alpha=0.5,**kwargs):
  x,y=riordina(x,y)
  if 'smussa' not in kwargs:
    kwargs['smussa']=1
  from scipy.interpolate import UnivariateSpline
  import logging
  import warnings
  logging.getLogger('matplotlib').setLevel(logging.ERROR)
  logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    f = UnivariateSpline(x, y , s=kwargs['smussa']/10000*0.5)
  x_new = np.linspace(min(x), max(x), 200)
  y_new = f(x_new)
  if 'smussa' in kwargs:
    del kwargs['smussa']
  plt.plot(x_new,y_new,**kwargs)
  if not 'color' in kwargs:
    kwargs['color']='skyblue'
  if riempi:
    y_limit = plt.ylim()
    plt.fill_between(x_new, y_new, y_limit[0],color=kwargs['color'],alpha=riempi_alpha)
  return

#STAMPA (print) ma piu figa (grazie chatgpt)
#       SINTASSI/ESEMPI: stampa(['ciaoo',20],['questo è un testo un po piu grande',50],['questo è piu piccolo ed è rosso',30,'red'])
#
#                        misura=40
#                        stampa([f'la tua misura è {misura}','blue'])

def stampa(frase, size=18, colore='inherit'):
    grandezza=size
    if isinstance(grandezza, str):
        colore, grandezza = grandezza, 18
    stile = f"style='font-size: {grandezza}px; display: inline-block; color: {colore}; white-space: pre;'"
    frase_formattata = frase.format(f=frase) if '{f}' in frase else frase
    testo_stili_multipli = f"<span {stile}>{frase_formattata}</span>"
    display(HTML(testo_stili_multipli))
  
def stampaplus(*frasi_grandezze_colore):
    color='inherit\''
    testo_stili_multipli = ""
    frase = ''
    for elemento in frasi_grandezze_colore:
        grandezza = 18
        colore=color
        if isinstance(elemento, (tuple, list)):
            if len(elemento) == 1:
                frase = elemento[0]
            if len(elemento) == 2:
                if isinstance(elemento[1], str):
                    frase, colore, grandezza = elemento[0], elemento[1], 18
                if isinstance(elemento[1], (int, float)):
                    frase, grandezza, colore = elemento[0], elemento[1], colore
            if len(elemento) == 3:
                if isinstance(elemento[1], str):
                    frase, colore, grandezza = elemento[0], elemento[1], elemento[2]
                elif isinstance(elemento[1], (int, float)):
                    frase, grandezza, colore = elemento[0], elemento[1], elemento[2]
        else:
            frase = str(elemento)

        stile = f"style='font-size: {grandezza}px; display: inline-block; color: {colore}; white-space: pre;'"
        frase_formattata = frase.format(f=frase) if '{f}' in frase else frase
        testo_stili_multipli += f"<span {stile}>{frase_formattata}</span>"

    display(HTML(testo_stili_multipli))





#CORRELAZIONE LINEAERE (Pearson):
def pearson(x,y):
  print('')
  print('')
  stampa(['Correlazione lineare di pearson:',25])
  import scipy as sp
  import pandas as pd
  pear=sp.stats.pearsonr(x,y)
  coeff=pear[0]
  s_coeff=((1-coeff**2)/(len(x)-2))**0.5
  tabella = {'correlazione_lineare:': coeff, 's_correlazione_lineare:': s_coeff}
  df = pd.Series(tabella)
  return df


#MEDIA
def media(*lista):
  if isinstance(lista[0],(int, float,np.int64,np.int32,np.float64,np.float32)):
    lis=[]
    for c in range (len(lista)):
      lis.append(lista[c])
  else:
    lis=list(lista[0])
  return sum(lis)/len(lis)


#DEVIAZIONE STANDARD (quella che ha nel denominatore n-1)
def std(*lista):
  import numpy as np
  if isinstance(lista[0],(int, float,np.int64,np.int32,np.float64,np.float32)):
    lis=[]
    for c in range (len(lista)):
      lis.append(lista[c])
  else:
    lis=list(lista[0])
  import numpy as np
  return np.std(lis,ddof=1)

#DEVIAZIONE STANDARD DELLA MEDIA
def stdmedia(*lista):
  import numpy as np
  if isinstance(lista[0],(int, float,np.int64,np.int32,np.float64,np.float32)):
    lis=[]
    for c in range (len(lista)):
      lis.append(lista[c])
  else:
    lis=list(lista[0])
  import numpy as np
  return (np.std(lis,ddof=1))/(len(lis))**0.5

#CHI QUADRO su una retta interpolante:  (BOZZA)
#               c'è bisogno di inserire i gradi di libertà utilizzando le keywords ddof= N-vincoli, non so se i gradi di libertà per una retta sono sempre N-2 nel caso poi modifico la funzione
#               se si assegna la funzione ad una variabile, alla variabile viene assegnato il p value risultante (VEDI ESEMPIO 2), oltre a stampare i dovuti risultati
           #SINTASSI: chi2retta(lista x,lista y, numero o lista incerteze y, variabile retta calcolata con la funzione fit, KEYWORDS)

           #KEYWORDS:

 #                   ddof= gradi di libertà
#                             imposta i gradi di libertà da utilizzare

#                    tabella=True
#                             mostra la tabella come le mostrava doro

           #ESEMPI:

#                ESEMPIO 1:
#                          chi2retta(x,y,sy,retta,ddof=(len(x)-2))
#
#                ESEMPIO 2:
#                          pvalue=chi2retta(x,y,sy,retta,ddof=(),tabella=True)



def chi2retta(x,y,sy,retta,p_value=False,**kwargs):
    from collections import namedtuple
    if hasattr(retta, '_fields'):
      retta = [[retta.intercetta,retta.s_intercetta],[retta.pendenza,retta.s_pendenza],retta.covarianza]
    from IPython.display import display
    if not 'ddof' in kwargs:
        kwargs['ddof']=len(x)-2
    if not 'tabella' in kwargs:
        kwargs['tabella']=False
    from scipy.stats import chi2
    import pandas as pd
    if isinstance (sy,(int,float)):
        incertezza=sy
        lista=[]
        for f in range (len(x)):
            lista.append(sy)
        sy=lista
    chi2singoli=[]
    attese=[]
    for c in range (len(x)):
        attesa=x[c]*retta[1][0]+retta[0][0]
        attese.append(attesa)
        chi2singolo=((y[c]-attesa)/sy[c])**2
        chi2singoli.append(chi2singolo)
    pvalue= round(1 - chi2.cdf(sum(chi2singoli), kwargs['ddof']),2)
    tabella=pd.DataFrame({'x_i':x,'y_i':y,'s_y':sy,'y_i*':attese,'chi2_i':chi2singoli,'chi2_m':sum(chi2singoli)})
    if kwargs['tabella']==True:
        stampa(['                                  tabella chi quadro     '])
        display(tabella)
    #stampa([f'il p-value del chi quadro è: {pvalue}',25])
    print('')
    if p_value:
      return pvalue
    return sum(chi2singoli)

def chi2(funzione,*args,p_value=False,mostra=False,**kwargs):
    args=list(args)
    if isinstance(args[0],(int, float,np.int64,np.int32,np.float64,np.float32)):
      args[0] = [args[0]]
    from IPython.display import display
    parametri=args[0]
    x=args[1]
    y=args[2]
    sy=args[3]
    if not 'ddof' in kwargs:
        kwargs['ddof']=len(x)-len(parametri)
    if not 'tabella' in kwargs:
        kwargs['tabella']=False
    from scipy.stats import chi2
    import pandas as pd
    if isinstance (sy,(int,float)):
        incertezza=sy
        lista=[]
        for f in range (len(x)):
            lista.append(sy)
        sy=lista
    chi2singoli=[]
    attese=[]
    for c in range (len(x)):
        attesa=funzione(x[c],*parametri)
        attese.append(attesa)
        chi2singolo=((y[c]-attesa)/sy[c])**2
        chi2singoli.append(chi2singolo)
    pvalue= round(1 - chi2.cdf(sum(chi2singoli), kwargs['ddof']),2)
    tabella=pd.DataFrame({'x_i':x,'y_i':y,'s_y':sy,'y_i*':attese,'chi2_i':chi2singoli,'chi2_m':sum(chi2singoli)})
    if kwargs['tabella']==True:
        stampa(['                                  tabella chi quadro     '])
        display(tabella)
        stampa([f'il p-value del chi quadro è: {pvalue}',35])
    if mostra:
        print('')
        stampa([f'il p-value del chi quadro è: {pvalue}',35])
        print('')
    if p_value:
      return pvalue
    return sum(chi2singoli)


def convfloat(lista):
    import numpy as np
    if isinstance(lista,(int, float,np.int64,np.int32,np.float64,np.float32)):
        lista=float(lista)
        return lista
    else:
        eh=[]
        for c in lista:
            eh.append(float(c))
        return np.array(eh)

#POTENZA: calcola la potenza di un numero o di una lista.
#         nel caso della lista significa che ritorna la stessa lista inserita ma con ogni suo elemento elevato al certo numero inserito.

#        SINTASSI: potenza(numero o lista , numero al quale elevare).

#         es: potenza([1,2,3],2) ---> [1,4,9]
#         es: potenza(2,3) ----> 8
#         es: potenza([2,4],-1) ----> [1/2, 1/4]
def potenza(a,b):
    a=convfloat(a)
    import numpy as np
    if isinstance(a,(int, float,np.int64,np.int32,np.float64,np.float32)):
        a=float(a)
        return a**b
    else:
        potenzalista=[]
        for c in a:
            c=float(c)
            potenzalista.append(c**b)
        return potenzalista


#MOLTIPLICA: moltiplica numeri con numeri, liste con liste e liste con numeri potendole/i inserire in numero e ordine arbitrario all interno dell'argomento
#            liste con liste significa che moltiplica il primo elemento della prima lista per il primo della seconda e poi il secondo della prima con il secondo della seconda ecc...

#        SINTASSI: moltiplica(numero o lista, numero o lista, numero o lista.......)

#        es: moltiplica([1,2,3],2)----> [2,4,6]
#        es: moltiplica(2,[1,2],[2,3])----> [4,12]
#        es: moltiplica([2,3],2,3)-----> [12,18]
#        es: moltiplica(2,4) ----> 8

def moltiplica(*argomento):
    import numpy as np
    conteggio=[]
    for h in range(0,len(argomento)):
        if isinstance(argomento[h],(int,float,np.int64,np.int32,np.float64,np.float32)):
            conteggio.append(1)
    if len(conteggio)==len(argomento):
        moltiplicazione=1
        for k in range(0,len(argomento)):
            moltiplicazione=moltiplicazione*argomento[k]
        return moltiplicazione
    else:

        args=list(argomento)
        for f in range(0, len(args)):
            if not isinstance(args[f],(int, float,np.int64,np.int32,np.float64,np.float32)) :
                lunghezza=len(args[f])
                indice=f

        for a in range(len(args)):
            if isinstance(args[a],(int, float,np.int64,np.int32,np.float64,np.float32)):
                lista=[]
                for e in range(0,lunghezza):
                    lista.append(args[a])
                args[a]=lista
        listarisultante=[1]*len(args[indice])
        for g in range(len(args)):
            args[g]=list(args[g])
        for b in range(0,len(args[0])):
            for c in range(0,len(args)):
                listarisultante[b]=listarisultante[b]*args[c][b]
        return listarisultante

#SOMMA: somma numeri con numeri, liste con liste e liste con numeri potendole/i inserire in numero e ordine arbitrario all interno dell'argomento
#       liste con liste significa che somma il primo elemento della prima lista per il primo della seconda e poi il secondo della prima con il secondo della seconda ecc..
#       nel caso si metta in argomento una singola lista, la funzione fornirà la somma degli elementi della lista

#         SINTASSI: somma(numero o lista, numero o lista, numero o lista.......)
#          es: somma([1,2,3])--->6
#          es: somma([1,2],[2,3]) ----> [3,5]
#          es: somma(2,[1,3],5) -----> [8,10]
#          es: somma(1,2)----> 3

def somma(*argomento):
    import numpy as np
    if len(argomento)==1 and not isinstance(argomento[0],(int, float,np.int64,np.int32,np.float64,np.float32)):
        return sum(argomento[0])
    conteggio=[]
    for h in range(0,len(argomento)):
        if isinstance(argomento[h],(int,float,np.int64,np.int32)):
            conteggio.append(1)
    if len(conteggio)==len(argomento):
        somma=0
        for k in range(0,len(argomento)):
            somma=somma+argomento[k]
        return somma
    else:
        args=list(argomento)
        for f in range(0, len(args)):
            if not isinstance(args[f], (int, float,np.int64,np.int32,np.float64,np.float32)):
                lunghezza=len(args[f])
                indice=f
        for a in range(len(args)):
            if isinstance(args[a],(int, float,np.int64,np.int32,np.float64,np.float32)):
                lista=[]
                for e in range(0,lunghezza):
                    lista.append(args[a])
                args[a]=lista
        listarisultante=[0]*len(args[indice])
        for b in range(0,len(args[0])):
            for c in range(0,len(args)):
                listarisultante[b]=listarisultante[b]+args[c][b]
        return listarisultante

def cifresignificative(number, uncertainty):
    # Calcola il numero di cifre significative per l'incertezza
    if uncertainty > 0:
        significant_digits = max(0, -int(math.floor(math.log10(uncertainty))))
    else:
        raise ValueError("L'incertezza deve essere maggiore di zero")
    
    # Formatta il numero e l'incertezza con il numero corretto di cifre significative
    number_str = f"{number:.{significant_digits}f}"
    uncertainty_str = f"_{{\\pm{uncertainty:.{significant_digits}f}}}"
    formatted_string = f"${number_str}{uncertainty_str}$"
    return formatted_string
def misurelatex(misure,incertezze):
    stringhe=[]
    if len(misure) != len(incertezze):
        raise ValueError('il numero di misure non corrisponde al numero d\'incertezze fornite, controlla che le due liste abbiano la stessa numerosità')
    for c in range (len(misure)):
      stringa=cifresignificative(misure[c],incertezze[c])
      stringhe.append(stringa)
    return stringhe

def residui(funzione,parametri,x,y,sy,colore='black',**kwargs):
  print('')
  if 'capsize' not in kwargs:
    kwargs['capsize']=2
  spazio=abs(min(x)-max(x))*0.08
  residui=[]
  for c in range (len(x)):
    residuo = y[c]-funzione(x[c],*parametri)
    residui.append(residuo)
  lex=np.array([min(x) - spazio,max(x) + spazio])
  plt.plot(lex,[0,0],color=colore)
  plt.errorbar(x,residui,yerr=sy,fmt='o',**kwargs)
  try:
    massim=max(sy)
  except:
    massim=sy
  plt.ylim(-max(np.abs(residui)*1.1+massim),max(np.abs(residui)*1.1+massim))
def combina_immagini(output_name="combined.png"):
    import os
    from PIL import Image
    # prendi tutti i file immagine nella cartella
    images = [f for f in os.listdir('.') 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) < 2:
        raise ValueError("Servono almeno due immagini nella cartella.")

    # ordina per data di modifica (dalla più vecchia alla più recente)
    images.sort(key=lambda x: os.path.getmtime(x))

    # penultima = prima immagine
    img1_path = images[-2]
    # ultima = seconda immagine
    img2_path = images[-1]

    # apri le immagini
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # dimensioni finali
    w = max(img1.width, img2.width)
    h = img1.height + img2.height

    # nuova immagine
    combined = Image.new("RGB", (w, h), (255, 255, 255))

    # incolla le due immagini
    combined.paste(img1, (0, 0))
    combined.paste(img2, (0, img1.height))

    combined.save(output_name)
    os.remove(img1_path)
    os.remove(img2_path)
    print(f"Creato file: {output_name}")
  
def fitlin(x,sx,y,sy,colorelinea=None,spessorelinea=None,**kwargs):
  import matplotlib.pyplot as plt
  '''
  1. ESEMPIO
  
    retta = fit(x, 0, y, sy, KEYWORDS...)
  
    pendenza = retta[1][0]
    incertezza_pendenza = retta[1][1] 
    intercetta = retta[0][0]
    incertezza_intercetta = retta[0][1] 
    covarianza_parametri = retta[2]

  2. KEYWORDS

          origine=True       ---> esegue un interpolazione passante per l origine
        
          plot=True          ---> plotta la retta
  
          residui=True       ---> plotta i residui 
          
      Se si vuole ridimensionare la retta (utile per plot=True):
          
          xdestra = valore   ---> l'estremo destro della retta, o meglio la coordinata x dell estremo destro della retta
          
          xsinistra = valore ---> l'estremo sinistro della retta, o meglio la coordinata x dell estremo sinistro della retta
      
      
      
      Modificare lo stile del plot e aggiungee delle label (utile per plot=True o residui=True):
          
          opzioniplot=True   ---> tutti gli argomenti che appaiono dopo questa keyword sono quelli per regolare il plot
                                  se si mette questa keyword dopo non si possono mettere altre keyword che non siano di matplotlib.
                                  cioè sono le opzioni di matplotlib... tipo color='blue' eccc... 

                                  
                                  sono le opzioni per plt.plot se hai fatto plot=True,
                                  
                                  sono le opzioni per plt.errorbar se hai fatto residui=True  (modifica i punti, colore delle errorbar, forma del punto ecc)
                                  
                                  Esempio
                                  fit(x, 0, y, sy, plot=True, opzioniplot=True, label='$y=ax+b$', linestyle='--')
                                  
                                  fit(x, 0, y, sy, residui=True, opzioniplot=True, label='punti sperimentali', capsize=4, ecolor='red')


                                  
                                  Se hai fatto residui = True puoi comunque variare il linestyle della retta al centro semplicemente scrivendo 
                                  la keyword per il linestyle (la funzione capisce che è una keyword che appartiene alla retta e non ai punti di plt.errorbar)


  3. Altro
    per ulteriori esempi e utilizzi esegui la riga:
    guida() '''

  opzioniplot={}
  boo=0
  for chiave, valore in kwargs.items():
    if boo==1:
      opzioniplot[chiave]=valore
    if chiave=='opzioniplot':
      boo=1
  if not 'origine' in kwargs:
    kwargs['origine']=False
  if not 'plot' in kwargs:
    kwargs['plot']=False
  if not 'residui' in kwargs:
    kwargs['residui']=False
  if not 'plot_veloce' in kwargs:
    kwargs['plot_veloce']=False
    
    
  N=len(x) #  Numerosità
  if len(x)!=len(y):
      raise ValueError('la numerosità delle misure in ascissa è diverse da quelle in ordinata')
      return 
  spazio=abs(min(x)-max(x))*0.08
  if 'xsinistra' in kwargs:
    spazio=0
  if 'xdestra' in kwargs:
    spazio=0
  if not 'xsinistra' in kwargs:
    kwargs['xsinistra']=min(x)
  if not 'xdestra' in kwargs:
    kwargs['xdestra']=max(x)
  if not isinstance (sy,(int,float)):
    sy=list(sy)
  if not isinstance(sx,(int,float)):
    sx=list(sx)
  def caso1(x,y,sy):
      x=convfloat(x)
      y=convfloat(y)
      sy=convfloat(sy)
      delta=N*somma(potenza(x,2))-(somma(x))**2
      intercetta=(1/delta)*(somma(potenza(x,2))*somma(y)-somma(x)*somma(moltiplica(x,y)))
      pendenza=(1/delta)*(N*somma(moltiplica(x,y))-somma(x)*somma(y))
      erroreintercetta=sy*((somma(potenza(x,2))/delta)**0.5)
      errorependenza=sy*((N/delta)**0.5)
      return [[intercetta,erroreintercetta],[pendenza,errorependenza]]
  def caso2(x,y,sy):
      x=convfloat(x)
      y=convfloat(y)
      sy=convfloat(sy)
      delta=somma(moltiplica(1,potenza(sy,-2)))*somma(moltiplica(potenza(x,2),potenza(sy,-2)))-(somma(moltiplica(x,potenza(sy,-2))))**2
      intercetta=(1/delta)*(somma(moltiplica(potenza(x,2),potenza(sy,-2)))*somma(moltiplica(y,potenza(sy,-2)))-somma(moltiplica(x,potenza(sy,-2)))*somma(moltiplica(x,y,potenza(sy,-2))))
      pendenza=(1/delta)*(somma(moltiplica(1,potenza(sy,-2)))*somma(moltiplica(x,y,potenza(sy,-2)))-somma(moltiplica(x,potenza(sy,-2)))*somma(moltiplica(y,potenza(sy,-2))))
      erroreintercetta=((1/delta)*somma(moltiplica(potenza(x,2),potenza(sy,-2))))**0.5
      errorependenza=((1/delta)*somma(moltiplica(1,potenza(sy,-2))))**0.5
      return [[intercetta,erroreintercetta],[pendenza,errorependenza]]
  def caso3(x,sx,y,sy):
      x=convfloat(x)
      y=convfloat(y)
      sy=convfloat(sy)
      sx=convfloat(sx)
      if isinstance (sy, (float, int)):
          retta=caso1(x,y,sy)
      if not isinstance (sy, (float, int)):
          retta=caso2(x,y,sy)
      b=retta[1][0]
      if isinstance(sy,(int, float)):
          incertezzasingola=sy
          sy=[]
          for c in range(0,len(x)):
              sy.append(incertezzasingola)
      if isinstance(sx,(int, float)):
          incertezzasingola=sx
          sx=[]
          for c in range(0,len(x)):
              sx.append(incertezzasingola)
      if len(sy) != len(x):
          raise ValueError('il numero di incertezze inserite per le misure in ordinata è diverso dalla numerosità delle misure')
      if len(sx) != len(x):
          raise ValueError('il numero di incertezze inserite per le misure in ascissa è diverso dalla numerosità delle misure')
      if len(sx) != len(x) or len(sy) != len(x):
          print('controlla i dati e riprova')
          return 
      si=potenza(somma(potenza(sy,2),moltiplica(b**2,potenza(sx,2))),0.5)
      delta=somma(moltiplica(1,potenza(si,-2)))*somma(moltiplica(potenza(x,2),potenza(si,-2)))-(somma(moltiplica(x,potenza(si,-2))))**2
      intercetta=(1/delta)*(somma(moltiplica(potenza(x,2),potenza(si,-2)))*somma(moltiplica(y,potenza(si,-2)))-somma(moltiplica(x,potenza(si,-2)))*somma(moltiplica(x,y,potenza(si,-2))))
      pendenza=(1/delta)*(somma(moltiplica(1,potenza(si,-2)))*somma(moltiplica(x,y,potenza(si,-2)))-somma(moltiplica(x,potenza(si,-2)))*somma(moltiplica(y,potenza(si,-2))))
      erroreintercetta=((1/delta)*somma(moltiplica(potenza(x,2),potenza(si,-2))))**0.5
      errorependenza=((1/delta)*somma(potenza(si,-2)))**0.5
      return [[intercetta,erroreintercetta],[pendenza,errorependenza]]
  if kwargs['origine']==True:
    if isinstance(sy,(int, float)):
        incertezzasingola=sy
        sy=[]
        for c in range(0,len(x)):
          sy.append(incertezzasingola)
    if sx==0:
      if isinstance(sx,(int, float)):
          incertezzasingola=sx
          sx=[]
          for c in range(0,len(x)):
              sx.append(incertezzasingola)
      from scipy.optimize import curve_fit
      def funzione(x,a):
        return x*a
      parametri, covarianza=curve_fit(funzione,x,y,sigma=sy,absolute_sigma=True)
      matrice = [[0,0],[parametri[0],covarianza[0][0]**0.5]]
    else:
      from scipy.odr import Model,RealData,ODR
      def funzione(parametri,x):
        return parametri[0]*x
      rettaa=caso3(x,sx,y,sy)
      modello=Model(funzione)
      data=RealData(x,y,sx=sx,sy=sy)
      odr = ODR(data, modello, beta0=[rettaa[1][0]])
      result = odr.run()
      matrice = [[0,0],[result.beta[0],result.sd_beta[0][0]]]


  elif sx == 0 and isinstance (sy, (float,int)):
      matrice = caso1(x,y,sy)

  elif sx == 0 and not isinstance (sy, (float,int)):
      if len(sy) != len(x):
          mess='''
          il numero di incertezze inserite per le misure in ordinata è diverso dalla numerosità delle misure.
          se le misure in ordinata hanno tutte la stessa incertezza è possibile insererire il singolo valore in argomento.
          '''
          raise ValueError(mess)
          return 
      matrice = caso2(x,y,sy)

  else:
      matrice = caso3(x,sx,y,sy)

  intercetta=matrice[0][0]
  erroreintercetta=matrice[0][1]
  pendenza=matrice[1][0]
  errorependenza=matrice[1][1]
  if kwargs['plot_veloce']==True:
    plt.figure(figsize=(8,5))
    lex=np.array([kwargs['xsinistra'] - spazio,kwargs['xdestra'] + spazio])
    ley=pendenza*lex + intercetta
    plt.plot(lex,ley,color=colorelinea,lw=spessorelinea)
    plt.errorbar(x,y,yerr=sy,xerr=sx,fmt='o',**opzioniplot)
    plt.legend()
    plt.figure(figsize=(8,3))
    if 'capsize' not in opzioniplot:
      opzioniplot['capsize']=2
    if 'linestyle' not in opzioniplot:
      opzioniplot['linestyle']='-'
    import matplotlib.pyplot as plt
    residui=[]
    for c in range (len(x)):
      residuo=y[c]-(pendenza*x[c]+intercetta)
      residui.append(residuo)
    lex=np.array([kwargs['xsinistra'] - spazio,kwargs['xdestra'] + spazio])
    plt.plot(lex,[0,0],linestyle=opzioniplot['linestyle'],color=colorelinea)
    if 'linestyle' in opzioniplot:
      del opzioniplot['linestyle']
    if 'label' in opzioniplot:
      del opzioniplot['label']
    plt.errorbar(x,residui,yerr=sy,fmt='o',**opzioniplot)
    try:
      massim=max(sy)
    except:
      massim=sy
    plt.ylim(-max(np.abs(residui)*1.1+massim),max(np.abs(residui)*1.1+massim))
    
  if kwargs['plot']==True:
    import matplotlib.pyplot as plt
    lex=np.array([kwargs['xsinistra'] - spazio,kwargs['xdestra'] + spazio])
    ley=pendenza*lex + intercetta
    plt.plot(lex,ley,**opzioniplot)
  if kwargs['residui']==True:
    if kwargs['plot']==True:
      plt.figure()
    if 'capsize' not in opzioniplot:
      opzioniplot['capsize']=2
    if 'linestyle' not in opzioniplot:
      opzioniplot['linestyle']='-'
    import matplotlib.pyplot as plt
    residui=[]
    for c in range (len(x)):
      residuo=y[c]-(pendenza*x[c]+intercetta)
      residui.append(residuo)
    lex=np.array([kwargs['xsinistra'] - spazio,kwargs['xdestra'] + spazio])
    plt.plot(lex,[0,0],linestyle=opzioniplot['linestyle'],color=colorelinea)
    if 'linestyle' in opzioniplot:
      del opzioniplot['linestyle']
    plt.errorbar(x,residui,yerr=sy,fmt='o',**opzioniplot)
    try:
      massim=max(sy)
    except:
      massim=sy
    plt.ylim(-max(np.abs(residui)*1.1+massim),max(np.abs(residui)*1.1+massim))
  from scipy.odr import RealData, Model, ODR
  def fun(param, x):
    return param[0] * x + param[1]
  if sx==0 and isinstance (sx,(int, float)): #se si trascurano le incertezze sulle x non va, allora le assumo piccolissime
    sx=1e-60
  if sy==0 and isinstance (sy,(int, float)): #se si trascurano le incertezze sulle x non va, allora le assumo piccolissime
    sy=1e-60
  model = Model(fun)
  data = RealData(x, y, sx=sx, sy=sy)
  odr = ODR(data, model, beta0=[pendenza,intercetta])
  resultt = odr.run()
  cov = resultt.cov_beta[1][0] 
  from collections import namedtuple
  matrice = namedtuple('parametri_dall_interpolazione', ['intercetta', 's_intercetta','pendenza','s_pendenza','covarianza'])
  matrice = matrice(intercetta = intercetta, s_intercetta=erroreintercetta,pendenza=pendenza,s_pendenza=errorependenza,covarianza=cov)
  return matrice



def fit(x,sx,y,sy,**kwargs):
  '''
  1. ESEMPIO
  
    retta = fit(x, 0, y, sy, KEYWORDS...)
  
    pendenza = retta[1][0]
    incertezza_pendenza = retta[1][1] 
    intercetta = retta[0][0]
    incertezza_intercetta = retta[0][1] 
    covarianza_parametri = retta[2]

  2. KEYWORDS

          origine=True       ---> esegue un interpolazione passante per l origine
        
          plot=True          ---> plotta la retta
  
          residui=True       ---> plotta i residui 
          
      Se si vuole ridimensionare la retta (utile per plot=True):
          
          xdestra = valore   ---> l'estremo destro della retta, o meglio la coordinata x dell estremo destro della retta
          
          xsinistra = valore ---> l'estremo sinistro della retta, o meglio la coordinata x dell estremo sinistro della retta
      
      
      
      Modificare lo stile del plot e aggiungee delle label (utile per plot=True o residui=True):
          
          opzioniplot=True   ---> tutti gli argomenti che appaiono dopo questa keyword sono quelli per regolare il plot
                                  se si mette questa keyword dopo non si possono mettere altre keyword che non siano di matplotlib.
                                  cioè sono le opzioni di matplotlib... tipo color='blue' eccc... 

                                  
                                  sono le opzioni per plt.plot se hai fatto plot=True,
                                  
                                  sono le opzioni per plt.errorbar se hai fatto residui=True  (modifica i punti, colore delle errorbar, forma del punto ecc)
                                  
                                  Esempio
                                  fit(x, 0, y, sy, plot=True, opzioniplot=True, label='$y=ax+b$', linestyle='--')
                                  
                                  fit(x, 0, y, sy, residui=True, opzioniplot=True, label='punti sperimentali', capsize=4, ecolor='red')


                                  
                                  Se hai fatto residui = True puoi comunque variare il linestyle della retta al centro semplicemente scrivendo 
                                  la keyword per il linestyle (la funzione capisce che è una keyword che appartiene alla retta e non ai punti di plt.errorbar)


  3. Altro
    per ulteriori esempi e utilizzi esegui la riga:
    guida() '''

  opzioniplot={}
  boo=0
  for chiave, valore in kwargs.items():
    if boo==1:
      opzioniplot[chiave]=valore
    if chiave=='opzioniplot':
      boo=1
  if not 'origine' in kwargs:
    kwargs['origine']=False
  if not 'plot' in kwargs:
    kwargs['plot']=False
  if not 'residui' in kwargs:
    kwargs['residui']=False
  N=len(x) #  Numerosità
  if len(x)!=len(y):
      raise ValueError('la numerosità delle misure in ascissa è diverse da quelle in ordinata')
      return 
  spazio=abs(min(x)-max(x))*0.08
  if 'xsinistra' in kwargs:
    spazio=0
  if 'xdestra' in kwargs:
    spazio=0
  if not 'xsinistra' in kwargs:
    kwargs['xsinistra']=min(x)
  if not 'xdestra' in kwargs:
    kwargs['xdestra']=max(x)
  if not isinstance (sy,(int,float)):
    sy=list(sy)
  if not isinstance(sx,(int,float)):
    sx=list(sx)
  def caso1(x,y,sy):
      delta=N*somma(potenza(x,2))-(somma(x))**2
      intercetta=(1/delta)*(somma(potenza(x,2))*somma(y)-somma(x)*somma(moltiplica(x,y)))
      pendenza=(1/delta)*(N*somma(moltiplica(x,y))-somma(x)*somma(y))
      erroreintercetta=sy*((somma(potenza(x,2))/delta)**0.5)
      errorependenza=sy*((N/delta)**0.5)
      return [[intercetta,erroreintercetta],[pendenza,errorependenza]]
  def caso2(x,y,sy):
      delta=somma(moltiplica(1,potenza(sy,-2)))*somma(moltiplica(potenza(x,2),potenza(sy,-2)))-(somma(moltiplica(x,potenza(sy,-2))))**2
      intercetta=(1/delta)*(somma(moltiplica(potenza(x,2),potenza(sy,-2)))*somma(moltiplica(y,potenza(sy,-2)))-somma(moltiplica(x,potenza(sy,-2)))*somma(moltiplica(x,y,potenza(sy,-2))))
      pendenza=(1/delta)*(somma(moltiplica(1,potenza(sy,-2)))*somma(moltiplica(x,y,potenza(sy,-2)))-somma(moltiplica(x,potenza(sy,-2)))*somma(moltiplica(y,potenza(sy,-2))))
      erroreintercetta=((1/delta)*somma(moltiplica(potenza(x,2),potenza(sy,-2))))**0.5
      errorependenza=((1/delta)*somma(moltiplica(1,potenza(sy,-2))))**0.5
      return [[intercetta,erroreintercetta],[pendenza,errorependenza]]
  def caso3(x,sx,y,sy):
      if isinstance (sy, (float, int)):
          retta=caso1(x,y,sy)
      if not isinstance (sy, (float, int)):
          retta=caso2(x,y,sy)
      b=retta[1][0]
      if isinstance(sy,(int, float)):
          incertezzasingola=sy
          sy=[]
          for c in range(0,len(x)):
              sy.append(incertezzasingola)
      if isinstance(sx,(int, float)):
          incertezzasingola=sx
          sx=[]
          for c in range(0,len(x)):
              sx.append(incertezzasingola)
      if len(sy) != len(x):
          raise ValueError('il numero di incertezze inserite per le misure in ordinata è diverso dalla numerosità delle misure')
      if len(sx) != len(x):
          raise ValueError('il numero di incertezze inserite per le misure in ascissa è diverso dalla numerosità delle misure')
      if len(sx) != len(x) or len(sy) != len(x):
          print('controlla i dati e riprova')
          return 
      si=potenza(somma(potenza(sy,2),moltiplica(b**2,potenza(sx,2))),0.5)
      delta=somma(moltiplica(1,potenza(si,-2)))*somma(moltiplica(potenza(x,2),potenza(si,-2)))-(somma(moltiplica(x,potenza(si,-2))))**2
      intercetta=(1/delta)*(somma(moltiplica(potenza(x,2),potenza(si,-2)))*somma(moltiplica(y,potenza(si,-2)))-somma(moltiplica(x,potenza(si,-2)))*somma(moltiplica(x,y,potenza(si,-2))))
      pendenza=(1/delta)*(somma(moltiplica(1,potenza(si,-2)))*somma(moltiplica(x,y,potenza(si,-2)))-somma(moltiplica(x,potenza(si,-2)))*somma(moltiplica(y,potenza(si,-2))))
      erroreintercetta=((1/delta)*somma(moltiplica(potenza(x,2),potenza(si,-2))))**0.5
      errorependenza=((1/delta)*somma(potenza(si,-2)))**0.5
      return [[intercetta,erroreintercetta],[pendenza,errorependenza]]
  if kwargs['origine']==True:
    if isinstance(sy,(int, float)):
        incertezzasingola=sy
        sy=[]
        for c in range(0,len(x)):
          sy.append(incertezzasingola)
    if sx==0:
      if isinstance(sx,(int, float)):
          incertezzasingola=sx
          sx=[]
          for c in range(0,len(x)):
              sx.append(incertezzasingola)
      from scipy.optimize import curve_fit
      def funzione(x,a):
        return x*a
      parametri, covarianza=curve_fit(funzione,x,y,sigma=sy,absolute_sigma=True)
      matrice = [[0,0],[parametri[0],covarianza[0][0]**0.5]]
    else:
      from scipy.odr import Model,RealData,ODR
      def funzione(parametri,x):
        return parametri[0]*x
      rettaa=caso3(x,sx,y,sy)
      modello=Model(funzione)
      data=RealData(x,y,sx=sx,sy=sy)
      odr = ODR(data, modello, beta0=[rettaa[1][0]])
      result = odr.run()
      matrice = [[0,0],[result.beta[0],result.sd_beta[0][0]]]


  elif sx == 0 and isinstance (sy, (float,int)):
      matrice = caso1(x,y,sy)

  elif sx == 0 and not isinstance (sy, (float,int)):
      if len(sy) != len(x):
          mess='''
          il numero di incertezze inserite per le misure in ordinata è diverso dalla numerosità delle misure.
          se le misure in ordinata hanno tutte la stessa incertezza è possibile insererire il singolo valore in argomento.
          '''
          raise ValueError(mess)
          return 
      matrice = caso2(x,y,sy)

  else:
      matrice = caso3(x,sx,y,sy)

  intercetta=matrice[0][0]
  erroreintercetta=matrice[0][1]
  pendenza=matrice[1][0]
  errorependenza=matrice[1][1]
  if kwargs['plot']==True:
    import matplotlib.pyplot as plt
    lex=np.array([kwargs['xsinistra'] - spazio,kwargs['xdestra'] + spazio])
    ley=pendenza*lex + intercetta
    plt.plot(lex,ley,**opzioniplot)
  if kwargs['residui']==True:
    if kwargs['plot']==True:
      plt.figure()
    if 'capsize' not in opzioniplot:
      opzioniplot['capsize']=2
    if 'linestyle' not in opzioniplot:
      opzioniplot['linestyle']='-'
    import matplotlib.pyplot as plt
    residui=[]
    for c in range (len(x)):
      residuo=y[c]-(pendenza*x[c]+intercetta)
      residui.append(residuo)
    lex=np.array([kwargs['xsinistra'] - spazio,kwargs['xdestra'] + spazio])
    plt.plot(lex,[0,0],linestyle=opzioniplot['linestyle'])
    if 'linestyle' in opzioniplot:
      del opzioniplot['linestyle']
    plt.errorbar(x,residui,yerr=sy,fmt='o',**opzioniplot)
    try:
      massim=max(sy)
    except:
      massim=sy
    plt.ylim(-max(np.abs(residui)*1.1+massim),max(np.abs(residui)*1.1+massim))
  from scipy.odr import RealData, Model, ODR
  def fun(param, x):
    return param[0] * x + param[1]
  if sx==0 and isinstance (sx,(int, float)): #se si trascurano le incertezze sulle x non va, allora le assumo piccolissime
    sx=1e-60
  if sy==0 and isinstance (sy,(int, float)): #se si trascurano le incertezze sulle x non va, allora le assumo piccolissime
    sy=1e-60
  model = Model(fun)
  data = RealData(x, y, sx=sx, sy=sy)
  odr = ODR(data, model, beta0=[pendenza,intercetta])
  resultt = odr.run()
  cov = resultt.cov_beta[1][0] 
  matrice = [[intercetta,erroreintercetta],[pendenza,errorependenza],cov]
  return matrice

#ERRORE A POSTERIORI per RETTA:
def posterioriretta(x,y,retta):
  x=np.array(x)
  attese=x*retta[1][0]+retta[0][0]
  return ((somma(potenza(somma(y,moltiplica(attese,-1)),2)))/(len(x)-2))**0.5


#ESPORTA SU EXCEL: metti in input una lista (lista di liste o una lista semplice), un array, una matrice ecc e le salva in un file excel.
#                  SINTASSI:  excel(lista,'nomefile')
#                      ESEMPIO:  excel (lista, 'ciao' )


def excel(lista, stringa,**kwargs):
  nomefile=stringa + '.xlsx'
  if 'transponi' not in kwargs:
    kwargs['transponi']=False
  if 'cifre_decimali' in kwargs:
    lista=np.round(lista,kwargs['cifre_decimali'])
  if isinstance (lista[0],(int, float)):
    df = pd.DataFrame(lista)
    if kwargs['transponi']==True:
      df = pd.DataFrame(lista).transpose()
    return  df.to_excel(nomefile, index=False)
  else:
    df = pd.DataFrame(lista).transpose()
    return  df.to_excel(nomefile, index=False)

def minimirelativi(lista,soglia=1,indici=False,contrario=False,fontsize=12,color='purple',**kwargs):
    import numpy as np
    indici=[]
    minimi=[]
    for c in range(soglia, len(lista)-soglia):
        if lista[c+1]>lista[c] and lista[c-1]>lista[c] and lista[c+soglia]>lista[c] and lista[c-soglia]>lista[c]:
            indici.append(c)
            minimi.append(lista[c])
    if contrario:
        indici.reverse()
        minimi.reverse()
    if 'plot' in kwargs:
        y,x = riordina (lista, list(kwargs['plot']), rispetto= list(kwargs['plot']))
        x=np.array(kwargs['plot'])
        y=np.array(lista)
        x1=x[indici]
        y1=y[indici]
        #plt.errorbar(x1,y1,yerr=max(y)*0.1,elinewidth=0.5,fmt='None',color=color)
        plt.plot(x,y)
        for c in range(len(x1)):
          testo=f'{c}'
          plt.text(x1[c],y1[c],testo,fontsize=fontsize,ha='center',va='top',color=color)
          plt.scatter(x1[c],y1[c],s=2,color=color)
    if indici:
        return np.array(indici)
    else:
        return minimi



def massimirelativi(lista,soglia=1,indici=False,contrario=False,fontsize=12,color='purple',**kwargs):
    import numpy as np
    indici=[]
    massimi=[]
    for c in range(soglia, len(lista)-soglia):
        if lista[c+1]<lista[c] and lista[c-1]<lista[c] and lista[c+soglia]<lista[c] and lista[c-soglia]<lista[c]:
            indici.append(c)
            massimi.append(lista[c])
    if contrario:
        indici.reverse()
        massimi.reverse()
    if 'plot' in kwargs:
        y,x = riordina (lista, list(kwargs['plot']), rispetto= list(kwargs['plot']))
        x=np.array(kwargs['plot'])
        y=np.array(lista)
        x1=x[indici]
        y1=y[indici]
        #plt.errorbar(x1,y1,yerr=max(y)*0.1,elinewidth=0.5,fmt='None',color=color)
        plt.plot(x,y)
        for c in range(len(x1)):
          testo=f'{c}'
          plt.text(x1[c],y1[c],testo,fontsize=fontsize,ha='center',va='bottom',color=color)
          plt.scatter(x1[c],y1[c],s=2,color=color)
    if indici:
        return np.array(indici)
    else:
        return massimi






#SUONA: fa partire un suono mettendo il link del download del suono in argomento
#   SINTASSI: suona('www.ciao.mp3')
#   trovare link suoni: trovate un suono e lo scaricate, dal browser: tasto destro nel file scaricato e fate copia link download (almeno da edge)
def suona(link):
  presente=False
  for c, d in suoni.items():
    if link==c:
      link=d
      presente=True
  elenco = ', '.join("'{0}'".format(key) for key in suoni.keys()) + '.'
  if link=='elenco':
    print('l\'elenco dei suoni disponibili è:', elenco)
    return
  if isinstance(link, str) and presente==False:
    raise ValueError('Questo suono non è stato ancora aggiunto, per visualizzare l\'elenco con tutti i suoni metti in argomento \'elenco\'')
    return
  link=linkdiretto(link)
  response = requests.get(link)
  audio_data = BytesIO(response.content)
  display(Audio(data=audio_data.read(), autoplay=True))
  
      


#MEDIA PESATA fra misure:
#                         è possibile  inserire in argomento le singole misure e incertezze o le 2 liste contenti le misure e le incertesze
#
#             COSA RITORNA: ritorna la seguente lista [mediapesata,incertezzamediapesata]
#             SINTASSI:
#                      singole misure:
#                                      mediapesata(x1,sx1,x2,sx2,x3,sx3....)
#
#                      liste:
#                                      mediapesata(x,sx)
#
#             ESEMPIO:
#                     b = mediapesata(x,sx)
#                     media_pesata = b[0]
#                     incertezza_media_pesata = b[1]

def mediapesata(*argo):
    arg=list(argo)
    if isinstance (arg[0],(int, float)):
        x=[]
        sx=[]
        for c in range (0,int(len(arg)),2):
            x.append(arg[c])
        for b in range (1,int(len(arg)),2):
            sx.append(arg[b])
        arg=[]
        arg.append(x)
        arg.append(sx)
    if isinstance (arg[1],(int,float)):
      sx=arg[1]
      lista=[]
      for h in range(0,len(arg[0])):
        lista.append(sx)
      arg[1]=lista
    k=somma(potenza(arg[1],-2))
    sx_mp=(1/k)**0.5
    x_mp=(1/k)*somma(moltiplica(arg[0],potenza(arg[1],-2)))
    return [x_mp,sx_mp]



#COMPATIBILITÀ           fra misure (con eventuale media pesata):
#
#              IN BREVE: calcola la compatibilità ed in caso positivo può farne la media pesata (in tal caso la soglia di compatibilità di default è 3) (vedi parte KEYWORDS!!!)
#                        è possibile inserire i dati sia in liste (o oggetti simili) che singolarmente (vedi parte SINTASSI)
#
#              SINTASSI:
#                        singolarmente:
#                                     compatibilità(x1,sx1,x2,sx2.....,keywords....)
#                        liste/array e oggetti simili:
#                                     compatibilità(x,sx,keywords...)
#
#              KEYWORDS: in argomento, una volta inserite le misure, sono disponibili varie keywords da poter utilizzare in ordine arbitrario.
#                        ESEMPIO KEYWORDS:
#                                         compatibilità(x,sx,soglia=2,mediapesata=True,mostratuttegruppo=True)
#
#                        mediapesata=True  --->  inserendola la funzione compatibilità ritornerà la media pesata ([mediapesata,erroremediapesata])
#                                                oltre a stamparne il risultato.
#                                                la media pesata viene calcolata tra le misure che hanno una compatibilità inferiore alla soglia
#                                                la soglia impostata di default è 3 ma puo essere cambiata con la specifica keyword
#
#                        soglia=valore_soglia_che_si_vuole ---> imposta il valore della soglia cioè quel valore per cui una misura
#                                                                 (due misure, dipende in che parte del processo si è...) viene scartata
#                                                                 dal calcolo della media pesata cioè se la sua compatibilità risulta essere maggiore della soglia
#
#                        mostratutte=True ---> mostra le singole compatibilità tra tutte le misure
#
#                        mostratutteconsoglia=True ---> mostra le singole compatibilità che sono sotto la soglia
#
#                        mostratuttegruppo=True ----> nel caso si inseriscano piu di 2 misure mostra le singole compatibilià all'interno del gruppo più numeroso di misure compatibili (vedi parte ECCEZIONI)
#
#                        mostragruppi=True ----> mostra tutti i gruppi di misure tra di loro compatibil
#
#               ECCEZIONI e approfondimenti:
#                        se in input inseriamo piu di 2 misure la funzione riporterà il gruppo più numeroso ('gruppo vincente') di misure fra di loro tutte compatibili (entro la soglia)
#                        e quindi eventualmente fare la media pesata di quelle misure (se mediapesata=True)
#                        capita non raramente però che ci siano due o più 'gruppi vincenti' cioè che hanno la stessa numerosità.
#                        in questo caso la funzione calcolerà il valore della compatibilità media per ogni gruppo e quindi sceglierà come vincente quella con il valore minore.
#                        poi da quest'ultimo verrà calcolata la media pesata
#                        in ogni caso verrano stampati tutti i gruppi vincenti con i loro errori nel caso possano essere utili.
def compatibilità(*argomento,**kargs):
    if not 'mostragruppi' in kargs:
        kargs['mostragruppi']=False
    if not 'mostratuttegruppo' in kargs:
        kargs['mostratuttegruppo']=False
    if not 'mediapesata' in kargs:
        kargs['mediapesata']=False
    if not 'mostratutte' in kargs:
        kargs['mostratutte']=False
    if not 'mostratutteconsoglia' in kargs:
        kargs['mostratutteconsoglia']=False
    if not 'soglia' in kargs:
        kargs['soglia']=3
    args=list(argomento)
    if isinstance (args[0],(int, float)):
        x=[]
        sx=[]
        for c in range (0,int(len(args)),2):
            x.append(args[c])
        for b in range (1,int(len(args)),2):
            sx.append(args[b])
        args=[]
        args.append(x)
        args.append(sx)
    if len(args[0]) != len(args[1]):
       return print('la lista delle misure e la lista dei loro errori hanno lunghezze diverse')
    if len(args[0])==2:
        c=(abs(args[0][0]-args[0][1]))/((args[1][0]**2+args[1][1]**2)**0.5)
        print('la compatibilità tra le due misure è:',c)
        if c < kargs['soglia'] and kargs['mediapesata']==True:
            mp=mediapesata(args[0],args[1])
            print('la loro media pesata è:', mp[0],'con incertezza sulla media pesata: ', mp[1])
        if c > kargs['soglia'] and kargs['mediapesata']==True:
            print('le due misure non sono compatibili con una la soglia di compatibilità pari a',kargs['soglia'],'quindi non è ragionevole effettuare una media pesata')
        return
    gruppi=[]
    gruppiincertezze=[]
    gruppicompatibilità=[]
    for r in range (0,len(args[0])):
        gruppi.append([])
        gruppiincertezze.append([])
    for f in range(0,len(args[0])):
        gruppi[f].append(args[0][f])
        gruppiincertezze[f].append(args[1][f])
        for d in range (0,len(args[0])):
            c=(abs(args[0][f]-args[0][d]))/((args[1][f]**2+args[1][d]**2)**0.5)
            if c != 0 and c < kargs['soglia']:
                gruppi[f].append(args[0][d])
                gruppiincertezze[f].append(args[1][d])
        for d in range (f,len(args[0])):
            c=(abs(args[0][f]-args[0][d]))/((args[1][f]**2+args[1][d]**2)**0.5)
            if c != 0 and kargs['mostratutte']==True:
                print('la compatibilità tra', args[0][f], 'e' ,args[0][d], 'è:',c)
            if c != 0 and kargs['mostratutteconsoglia']==True and c < kargs['soglia']:
                print('la compatibilità tra', args[0][f], 'e' ,args[0][d], 'è:',c)
    import copy

    copiagruppi=copy.deepcopy(gruppi)
    copiagruppiincertezze=copy.deepcopy(gruppiincertezze)
    for f in range(0,len(copiagruppi)):
        for u in range (1,len(copiagruppi[f])):
            for s in range (2,(len(copiagruppi[f]))):
                c=(abs(copiagruppi[f][u]-copiagruppi[f][s]))/((copiagruppiincertezze[f][u]**2+copiagruppiincertezze[f][s]**2)**0.5)
                if c > kargs['soglia']:
                    gruppi[f][u]=0
                    gruppi[f][s]=0
                    gruppiincertezze[f][u]=0
                    gruppiincertezze[f][s]=0
        while 0 in gruppi[f]:
            gruppi[f].remove(0)
        while 0 in gruppiincertezze[f]:
            gruppiincertezze[f].remove(0)
    sommegruppi=[]
    indici2=[]
    for e in range(0,len(gruppi)):
        sommegruppi.append(sum(gruppi[e]))
    for h in range(0,len(sommegruppi)):
        for g in range (h+1,len(sommegruppi)):
            if sommegruppi[g]==sommegruppi[h]:
                indici2.append(g)
    for d in indici2:
        gruppi[d]=0
        gruppiincertezze[d]=0
    while 0 in gruppi:
        gruppi.remove(0)
    while 0 in gruppiincertezze:
        gruppiincertezze.remove(0)
    if kargs['mostragruppi']==True:
        print(gruppi)
    maggiornumerosità=0
    indicegruppovincente=0
    gruppinumerositàuguale=[]
    gruppinumerositàuguale.append(0)

    for g in range(len(gruppi)):
        if len(gruppi[g]) == maggiornumerosità and len(gruppi[g]) != 1:
            gruppinumerositàuguale.append(len(gruppi[g]))
            comp1=[]
            compvincente=[]
            for o in range(0,len(gruppi[g])):
                for r in range (o+1,len(gruppi[g])):
                    c=(abs(gruppi[g][o]-gruppi[g][r]))/((gruppiincertezze[g][o]**2+gruppiincertezze[g][r]**2)**0.5)
                    comp1.append(c)
            for o in range(0,len(gruppi[indicegruppovincente])):
                for r in range (o+1,len(gruppi[indicegruppovincente])):
                    c=(abs(gruppi[indicegruppovincente][o]-gruppi[indicegruppovincente][r]))/((gruppiincertezze[indicegruppovincente][o]**2+gruppiincertezze[indicegruppovincente][r]**2)**0.5)
                    compvincente.append(c)
            if sum(comp1)/len(comp1) < sum(compvincente)/len(compvincente):
                maggiornumerosità=len(gruppi[g])
                indicegruppovincente=g
                compvincente=comp1
                gruppinumerositàuguale.pop()
        if len(gruppi[g]) > maggiornumerosità:
            maggiornumerosità=len(gruppi[g])
            indicegruppovincente=g
    gruppiNuguale=[]
    gruppiincertezzeNuguale=[]
    for y in range(0,len(gruppi)):
        if len(gruppi[y])==maggiornumerosità:
            gruppiNuguale.append(gruppi[y])
            gruppiincertezzeNuguale.append(gruppiincertezze[y])
    if maggiornumerosità==1:
        print('non sono state trovate misure compatibili sotto la soglia di', kargs['soglia'])
        return
    if max(gruppinumerositàuguale)==len(gruppi[indicegruppovincente]):
        print('non si è trovato un gruppo di misure fra di loro compatibili con una numerosità maggiore rispetto a tutti gli altri')
        print('cioè, fra i gruppi con numerosità maggiore di tutti gli altri, due o più gruppi hanno numerosità uguale')
        print('tali gruppi sono',gruppiNuguale)
        print('con le rispettive incertezze',gruppiincertezzeNuguale)
        print('')
        print('tra questi è stato scelto il gruppo il quale valore medio delle compatibilità fosse minore (quindi migliore)')
        print('rispetto a quelle degli altri gruppi mostrati:')
        print('')
        print('il gruppo di misure in questione è',gruppi[indicegruppovincente],'con misure compatibili fra loro sotto la soglia di',kargs['soglia'])
        print('la compatibilità media fra le misure del gruppo è:',sum(compvincente)/(len(compvincente)))
        print('')
        print('')
    else:
        comp2=[]
        for o in range(0,len(gruppi[indicegruppovincente])):
            for r in range (o+1,len(gruppi[indicegruppovincente])):
                c=(abs(gruppi[indicegruppovincente][o]-gruppi[indicegruppovincente][r]))/((gruppiincertezze[indicegruppovincente][o]**2+gruppiincertezze[indicegruppovincente][r]**2)**0.5)
                comp2.append(c)
        print('il gruppo più numeroso di misure compatibili tra loro è:',gruppi[indicegruppovincente], 'tutte con una compatibilità inferiore al valore soglia', kargs['soglia'])
        print('la media delle compatibilità fra le misure del gruppo è:',sum(comp2)/(len(comp2)))
    if kargs['mostratuttegruppo']==True:
        for t in range (0,len(gruppi[indicegruppovincente])):
            for r in range (t+1,len(gruppi[indicegruppovincente])):
                c=(abs(gruppi[indicegruppovincente][t]-gruppi[indicegruppovincente][r]))/((gruppiincertezze[indicegruppovincente][t]**2+gruppiincertezze[indicegruppovincente][r]**2)**0.5)
                print('la compatibilità tra',gruppi[indicegruppovincente][t],'e',gruppi[indicegruppovincente][r],'è:',c)
    if kargs['mediapesata']==True:
        mp=mediapesata(gruppi[indicegruppovincente],gruppiincertezze[indicegruppovincente])
        print('')
        print('la media pesata delle misure nel gruppo è',mp[0],'con incertezza sulla media pesata pari a:',mp[1])
        return([mp[0],mp[1]])




# type: ignore
