# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 14:00:26 2025

@author: carbse_ra10
"""

# app.py
import os, json, time
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from flask import (
    Flask, render_template, request,
    send_file, jsonify, abort
)
from flask_caching import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# ── Paths ──
BASE_DIR        = os.path.dirname(__file__)
STATIC_DIR      = os.path.join(BASE_DIR, 'static')
DATA_DIR        = os.path.join(STATIC_DIR, 'data')
IMAGE_DIR       = os.path.join(STATIC_DIR, 'images')
FIRST_STAGE_DIR = IMAGE_DIR               # <-- india_map.jpg lives here
PROFILE_DIR     = os.path.join(IMAGE_DIR, 'city-profile-for-single-city')
STATE_CITY_FILE = os.path.join(DATA_DIR, 'WeathertoolLocations.xlsx')
ADAPTIVE_FILE   = os.path.join(DATA_DIR, 'AdaptiveModels.xlsx')
BASE_MAP_FILE   = 'india_map.jpg'

# ── Load city metadata ──
_df = pd.read_excel(STATE_CITY_FILE, usecols=["State","City","Lat","Long"]).drop_duplicates('City')
STATE_CITY_MAP = _df.groupby('State')['City'].apply(lambda s: sorted(s)).to_dict()
CITY_META      = _df.set_index('City')[['Lat','Long']].to_dict('index')

# ── Load comfort models & typologies ──
COMFORT_MODELS = {}
TYP_TO_MODELS  = {}
try:
    xl = pd.read_excel(ADAPTIVE_FILE, sheet_name='Models', dtype=str)
    for _,r in xl.iterrows():
        m = r['Model'].strip()
        f = r['DataFilename'].strip()
        COMFORT_MODELS[m] = f
        for t in (r.get('Typologies','') or '').split(','):
            t = t.strip()
            if t: TYP_TO_MODELS.setdefault(t,[]).append(m)
except:
    # fallback if file missing/corrupt
    COMFORT_MODELS = {
        'ASHRAE 55':'City_Data_ASHRAE55.xlsx',
        'IMAC for Commercial':'City_Data_IMAC.xlsx',
        'IMAC for Residential':'City_Data_IMACR.xlsx'
    }
    TYP_TO_MODELS = {
        'Residential': list(COMFORT_MODELS),
        'Commercial':  list(COMFORT_MODELS)
    }
BUILDING_TYPOLOGIES = sorted(TYP_TO_MODELS)

# ── Month map ──
MONTH_MAP = {
    m: i+1 for i,m in enumerate([
        'Jan','Feb','Mar','Apr','May','Jun',
        'Jul','Aug','Sep','Oct','Nov','Dec'
    ])
}

# ── Helpers ──
def get_weather_file(model):
    fn = COMFORT_MODELS.get(model) or next(iter(COMFORT_MODELS.values()))
    return os.path.join(DATA_DIR, fn)

def load_sheet(sheet, model):
    try:
        return pd.read_excel(get_weather_file(model), sheet_name=sheet)
    except:
        return pd.DataFrame()

def get_min_from_excel(city, model):
    df = load_sheet('sheet_min', model)
    if 'City' not in df or 'sheet_min' not in df:
        return None
    row = df[df['City'].str.lower()==city.lower()]
    return float(row['sheet_min'].iloc[0]) if not row.empty else None

def get_max_from_excel(city, model):
    df = load_sheet('sheet_max', model)
    if 'City' not in df or 'sheet_max' not in df:
        return None
    row = df[df['City'].str.lower()==city.lower()]
    return float(row['sheet_max'].iloc[0]) if not row.empty else None

# ── Map composite ──
MIN_LAT,MAX_LAT = 6.0,36.0
MIN_LON,MAX_LON = 68.0,97.0
ICON_PATH       = os.path.join(STATIC_DIR,'icons','map_pin.png')
PIN_SIZE        = (24,24)

def latlon_to_pixel(lat,lon,W,H):
    x = (lon-MIN_LON)/(MAX_LON-MIN_LON)*W
    y = (MAX_LAT-lat)/(MAX_LAT-MIN_LAT)*H
    return int(round(x)),int(round(y))

def composite_multi(cities,colors):
    base = Image.open(os.path.join(FIRST_STAGE_DIR, BASE_MAP_FILE)).convert('RGBA')
    W,H  = base.size
    overlay = Image.new('RGBA',(W,H),(0,0,0,0))
    try:
        pin = Image.open(ICON_PATH).convert('RGBA').resize(PIN_SIZE,Image.LANCZOS)
    except:
        pin = None

    for city,color in zip(cities,colors):
        meta = CITY_META.get(city.title())
        if not meta:
            continue
        x,y = latlon_to_pixel(meta['Lat'], meta['Long'], W,H)
        if pin:
            tint = Image.new('RGBA', PIN_SIZE, color+(0,))
            icon = Image.alpha_composite(pin, tint)
            overlay.paste(icon, (x-PIN_SIZE[0]//2, y-PIN_SIZE[1]//2), icon)
        else:
            ImageDraw.Draw(overlay).ellipse([(x-6,y-6),(x+6,y+6)], fill=color+(255,))

    return Image.alpha_composite(base, overlay)



def plot_comfort_bands(df, kind='band', title='', min_y=None, max_y=None):
    df = df.copy()
    print("generate_chart input columns:", df.columns.tolist())
    print("Dataframe shape:", df.shape)
    
    if df.empty or 'Month' not in df.columns:
        # Nothing to plot
        buf = BytesIO()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        ax.axis('off')
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    # Month mapping
    if df['Month'].dtype == object:
        df['Mnum'] = df['Month'].map(MONTH_MAP)
    else:
        df['Mnum'] = df['Month'].astype(int)
    df = df.sort_values('Mnum')

    # Colors
    colors = {
        'Too Cold (< 80% Acceptable)': "#007f9c",
        'Cold (80-90% Acceptable)': "#0096c7",
        'Comfortable (90% Acceptable)': "#b0b0b0",
        'Warm (80-90% Acceptable)': "#ff7f71",
        'Too Hot (< 80% Acceptable)': "#e63946"
    }

    alpha_fill = 0.45
    fig, ax = plt.subplots(figsize=(9, 4))

    if kind == 'band':
        # Fill from bottom up
        ax.fill_between(
            df['Mnum'], min_y, df['Too Cold (< 80% Acceptable)'],
            color=colors['Too Cold (< 80% Acceptable)'], alpha=alpha_fill,
            label='Too Cold (< 80% Acceptable)'
        )
        ax.fill_between(
            df['Mnum'], df['Too Cold (< 80% Acceptable)'], df['Cold (80-90% Acceptable)'],
            color=colors['Cold (80-90% Acceptable)'], alpha=alpha_fill,
            label='Cold (80-90% Acceptable)'
        )
        ax.fill_between(
            df['Mnum'], df['Cold (80-90% Acceptable)'], df['Comfortable (90% Acceptable)'],
            color=colors['Comfortable (90% Acceptable)'], alpha=alpha_fill,
            label='Comfortable (90% Acceptable)'
        )
        ax.fill_between(
            df['Mnum'], df['Comfortable (90% Acceptable)'], df['Warm (80-90% Acceptable)'],
            color=colors['Warm (80-90% Acceptable)'], alpha=alpha_fill,
            label='Warm (80-90% Acceptable)'
        )
        ax.fill_between(
            df['Mnum'], df['Warm (80-90% Acceptable)'], df['Too Hot (< 80% Acceptable)'],
            color=colors['Too Hot (< 80% Acceptable)'], alpha=alpha_fill,
            label='Too Hot (< 80% Acceptable)'
        )
        # Running mean line
        if 'Toutdoorm' in df.columns:
            ax.plot(df['Mnum'], df['Toutdoorm'], 'k--', linewidth=1.5, label='30 Day Running Mean')
        ax.set_ylabel('Operative Temperature (°C)')
        if max_y is not None:
            ax.set_ylim(bottom=min_y, top=max_y)

    elif kind == 'stacked':
        cols = list(colors.keys())
        df_stack = df.copy()
        df_stack[cols] = df_stack[cols].div(df_stack[cols].sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(df_stack))
        for col in cols:
            ax.bar(
                df_stack['Mnum'], df_stack[col],
                bottom=bottom, color=colors[col],
                alpha=alpha_fill, label=col
            )
            bottom += df_stack[col].values
        ax.set_ylabel('% of Hours')
        ax.set_ylim(0, 100)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(list(MONTH_MAP.keys()))
    ax.set_xlabel('Month')
    ax.set_title(title)
    ax.grid(linestyle='--', alpha=0.5)
    ax.legend(
        fontsize='small',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False
    )

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf



# ── Comfort-hours plot ──
def plot_comfort_hours(df, city1, city2):
    # Normalize column names
    def find_col(city,kw):
        for c in df.columns:
            if city.lower() in c.lower() and kw.lower() in c.lower():
                return c
        return None

    c1_tot  = find_col(city1,'total')
    c1_comf = find_col(city1,'comfortable')
    c2_tot  = find_col(city2,'total')
    c2_comf = find_col(city2,'comfortable')
    if not all([c1_tot,c1_comf,c2_tot,c2_comf]):
        fig,ax=plt.subplots(figsize=(5,2))
        ax.text(0.5,0.5,"Data missing for hours",ha='center')
        ax.axis('off')
        buf=BytesIO(); fig.savefig(buf,'png',bbox_inches='tight'); buf.seek(0); plt.close(fig)
        return buf

    df2 = df.copy()
    df2['Temperature'] = pd.to_numeric(df2['Temperature'],errors='coerce')
    for col in (c1_tot,c1_comf,c2_tot,c2_comf):
        df2[col] = pd.to_numeric(df2[col],errors='coerce').fillna(0)

    df2 = df2.sort_values('Temperature')
    mask = (df2[c1_tot]>0)|(df2[c2_tot]>0)
    sub  = df2[mask] if mask.any() else df2
    tmin,tmax = sub['Temperature'].min(), sub['Temperature'].max()
    pad = 0.05*(tmax-tmin) if tmax>tmin else 1

    fig,ax=plt.subplots(figsize=(10,4))
    ax.fill_between(sub['Temperature'], sub[c1_comf], alpha=0.5, label=f"{city1} – Comfortable")
    ax.fill_between(sub['Temperature'], sub[c1_tot],  alpha=0.3, label=f"{city1} – Total")
    ax.fill_between(sub['Temperature'], sub[c2_comf], alpha=0.5, label=f"{city2} – Comfortable")
    ax.fill_between(sub['Temperature'], sub[c2_tot],  alpha=0.3, label=f"{city2} – Total")
    ax.set_xlim(tmin-pad, tmax+pad)
    ax.set_xticks(np.linspace(tmin,tmax,6))
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Operating Hours")
    ax.grid('--',alpha=0.5)
    ax.legend(frameon=True,fontsize='small')
    plt.tight_layout()

    buf=BytesIO(); fig.savefig(buf,'png',bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return buf

# ── Routes ──

@app.route('/')
def home():
    return render_template('home.html',
        states=sorted(STATE_CITY_MAP),
        typologies=BUILDING_TYPOLOGIES,
        typ_to_models=json.dumps(TYP_TO_MODELS),
        city_meta_json=json.dumps(CITY_META),
        ts=int(time.time())
    )

@app.route('/api/cities')
def api_cities():
    s = request.args.get('state','')
    return jsonify({'cities': STATE_CITY_MAP.get(s,[])})

@app.route('/map_preview')
@cache.cached(timeout=3600,query_string=True)
def map_preview():
    c1 = request.args.get('city1','')
    c2 = request.args.get('city2','')
    img = composite_multi([c1,c2], [(229,63,70),(0,127,156)])
    buf=BytesIO(); img.save(buf,'PNG'); buf.seek(0)
    return send_file(buf,mimetype='image/png')

@app.route('/analyze', methods=['POST'])
def analyze():
    typ   = request.form['typology']
    model = request.form['model']
    c1    = request.form['city1']
    c2    = request.form['city2']
    dfm   = load_sheet('MeanMaxMin', model)
    def mk(city):
        sub = dfm[dfm['City'].str.lower()==city.lower()]
        sub = sub[['Particular']+list(MONTH_MAP)].copy()
        sub.columns = ['Particular/Month'] + list(MONTH_MAP)
        sub[list(MONTH_MAP)] = sub[list(MONTH_MAP)].round(2)
        return sub.to_html(index=False, classes='table table-striped',border=0, float_format="%.2f")
    return jsonify(table1=mk(c1), table2=mk(c2))

@app.route('/chart/comfort1')
@cache.cached(timeout=3600,query_string=True)
def chart1():
    city  = request.args.get('city1','')
    model = request.args.get('model','')
    df    = load_sheet('RawData', model)
    sub   = df[df['City'].str.lower()==city.lower()]
    min_y, max_y = get_min_from_excel(city, model), get_max_from_excel(city, model)
    # --- pass via keywords, so kind='band' and title=city ----
    buf   = plot_comfort_bands(sub,
                                kind='band',
                                title=city,
                                min_y=min_y,
                                max_y=max_y)
    return send_file(buf, mimetype='image/png')

@app.route('/chart/comfort2')
@cache.cached(timeout=3600,query_string=True)
def chart2():
    city  = request.args.get('city2','')
    model = request.args.get('model','')
    df    = load_sheet('RawData', model)
    sub   = df[df['City'].str.lower()==city.lower()]
    min_y, max_y = get_min_from_excel(city, model), get_max_from_excel(city, model)
    buf   = plot_comfort_bands(sub,
                                kind='band',
                                title=city,
                                min_y=min_y,
                                max_y=max_y)
    return send_file(buf, mimetype='image/png')

@app.route('/chart/comfort_hours')
@cache.cached(timeout=3600,query_string=True)
def chart_hours():
    c1    = request.args.get('city1','')
    c2    = request.args.get('city2','')
    model = request.args.get('model','')
    df    = load_sheet('RawData8760', model)
    buf   = plot_comfort_hours(df, c1, c2)
    return send_file(buf,mimetype='image/png')




@app.route('/download/stats1')
def download_stats1():
    city1 = request.args.get('city1',''); model = request.args.get('model','')
    dfm = load_sheet('MeanMaxMin', model)
    sub = dfm[dfm['City'].str.lower()==city1.lower()]
    sub = sub[['Particular']+list(MONTH_MAP)].copy()
    sub.columns = ['Particular/Month']+list(MONTH_MAP)
    csv = sub.to_csv(index=False)
    return Response(csv, mimetype='text/csv', headers={'Content-Disposition':f'attachment;filename={city1}_stats.csv'})

@app.route('/download/stats2')
def download_stats2():
    city2 = request.args.get('city2',''); model = request.args.get('model','')
    dfm = load_sheet('MeanMaxMin', model)
    sub = dfm[dfm['City'].str.lower()==city2.lower()]
    sub = sub[['Particular']+list(MONTH_MAP)].copy()
    sub.columns = ['Particular/Month']+list(MONTH_MAP)
    csv = sub.to_csv(index=False)
    return Response(csv, mimetype='text/csv', headers={'Content-Disposition':f'attachment;filename={city2}_stats.csv'})

...# Download chart images
@app.route('/download/chart1')
def download_chart1():
    city = request.args.get('city1','')
    model = request.args.get('model','')
    buf = plot_comfort_bands(
        load_sheet('RawData',model).query("City.str.lower()=='{}'".format(city.lower())),
        kind='band', title=city,
        min_y=get_min_from_excel(city,model),
        max_y=get_max_from_excel(city,model)
    )
    return send_file(buf,
                     mimetype='image/png',
                     as_attachment=True,
                     download_name=f'{city}_comfort_bands.png')

@app.route('/download/chart2')
def download_chart2():
    city = request.args.get('city2','')
    model = request.args.get('model','')
    buf = plot_comfort_bands(
        load_sheet('RawData',model).query("City.str.lower()=='{}'".format(city.lower())),
        kind='band', title=city,
        min_y=get_min_from_excel(city,model),
        max_y=get_max_from_excel(city,model)
    )
    return send_file(buf,
                     mimetype='image/png',
                     as_attachment=True,
                     download_name=f'{city}_comfort_bands.png')

@app.route('/download/chart_hours')
def download_chart_hours():
    c1 = request.args.get('city1',''); c2 = request.args.get('city2','')
    model = request.args.get('model','')
    df = load_sheet('RawData8760', model)
    buf = plot_comfort_hours(df, c1, c2)
    return send_file(buf,
                     mimetype='image/png',
                     as_attachment=True,
                     download_name=f'{c1}_{c2}_comfort_hours.png')

if __name__=='__main__':
    app.run(debug=True,use_reloader=False)
