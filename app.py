
import os, json, time
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import joblib


from PIL import Image, ImageDraw
from flask import (
    Flask, render_template, request,
    send_file, jsonify, abort, Response
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
BASE_MAP_FILE   = os.path.join(IMAGE_DIR, 'city-profile-for-single-city/india-map.jpg')


# ── Load city metadata ──
_df = pd.read_excel(STATE_CITY_FILE).drop_duplicates('City')

# Sanitize column names
_df.columns = _df.columns.str.strip().str.replace('\u200b', '').str.replace(' ', '').str.title()

# Confirm column names if needed
print(_df.columns)


STATE_CITY_MAP = _df.groupby('State')['City'].apply(lambda s: sorted(s)).to_dict()
CITY_META = _df.set_index('City')[['X', 'Y']].to_dict('index')  # ✅ Use X/Y instead of Lat/Long


# Load once at top
# model_x, model_y = joblib.load(os.path.join(BASE_DIR, 'map_calibration/map_model.pkl'))

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
            if t:
                TYP_TO_MODELS.setdefault(t, []).append(m)
except Exception:
    COMFORT_MODELS = {
        'ASHRAE 55':'City_Data_ASHRAE55.xlsx',
        'IMAC for Commercial':'City_Data_IMAC.xlsx',
        'IMAC for Residential':'City_Data_IMACR.xlsx'
    }
    TYP_TO_MODELS = {
        'Residential': list(COMFORT_MODELS),
        'Commercial': list(COMFORT_MODELS)
    }
BUILDING_TYPOLOGIES = sorted(TYP_TO_MODELS)

# ── Month map ──
MONTH_MAP = {m: i+1 for i,m in enumerate([
    'Jan','Feb','Mar','Apr','May','Jun',
    'Jul','Aug','Sep','Oct','Nov','Dec'
])}

# ── Helpers ──

def get_weather_file(model):
    fn = COMFORT_MODELS.get(model) or next(iter(COMFORT_MODELS.values()))
    return os.path.join(DATA_DIR, fn)


def load_sheet(sheet, model):
    try:
        return pd.read_excel(get_weather_file(model), sheet_name=sheet)
    except Exception:
        return pd.DataFrame()


def get_min_from_excel(city, model):
    df = load_sheet('sheet_min', model)
    if 'City' not in df or 'sheet_min' not in df:
        return None
    row = df[df['City'].str.lower() == city.lower()]
    return float(row['sheet_min'].iloc[0]) if not row.empty else None


def get_max_from_excel(city, model):
    df = load_sheet('sheet_max', model)
    if 'City' not in df or 'sheet_max' not in df:
        return None
    row = df[df['City'].str.lower() == city.lower()]
    return float(row['sheet_max'].iloc[0]) if not row.empty else None

# ── Map composite ──
MIN_LAT, MAX_LAT = 6.0, 36.0
MIN_LON, MAX_LON = 68.0, 97.0
ICON_PATH        = os.path.join(STATIC_DIR, 'icons', 'map_pin.png')
PIN_SIZE         = (24, 24)

# def calibrated_latlon_to_pixel(lat, lon):
#     x = int(round(model_x.predict([[lat, lon]])[0]))
#     y = int(round(model_y.predict([[lat, lon]])[0]))
#     return x, y

def composite_multi(cities, colors):
    base = Image.open(os.path.join(FIRST_STAGE_DIR, BASE_MAP_FILE)).convert('RGBA')
    W, H = base.size
    overlay = Image.new('RGBA', (W, H), (0,0,0,0))
    try:
        pin = Image.open(ICON_PATH).convert('RGBA').resize(PIN_SIZE, Image.LANCZOS)
    except Exception:
        pin = None
    for city, color in zip(cities, colors):
        meta = CITY_META.get(city.title())
        if not meta:
            continue
        x, y = int(meta['X']), int(meta['Y'])  # ✅ Use direct x/y
        if pin:
            tint = Image.new('RGBA', PIN_SIZE, color + (0,))
            icon = Image.alpha_composite(pin, tint)
            overlay.paste(icon, (x - PIN_SIZE[0] // 2, y - PIN_SIZE[1] // 2), icon)
        else:
            ImageDraw.Draw(overlay).ellipse([(x-6, y-6), (x+6, y+6)], fill=color + (255,))
    return Image.alpha_composite(base, overlay)


# ── Plotting ──

# ── Plot bands (no legend) ──
def plot_comfort_bands(df,kind='band',title='',min_y=None,max_y=None):
    df=df.copy();
    if df.empty or 'Month' not in df: return _empty_png(8,4)
    df['Mnum']=df['Month'].map(MONTH_MAP) if df['Month'].dtype=='O' else df['Month'].astype(int)
    df=df.sort_values('Mnum')
    colors={
        'Too Cold (< 80% Acceptable)':'#007f9c',
        'Cold (80-90% Acceptable)':'#0096c7',
        'Comfortable (90% Acceptable)':'#b0b0b0',
        'Warm (80-90% Acceptable)':'#ff7f71',
        'Too Hot (< 80% Acceptable)':'#e63946'
    }
    alpha=0.45
    fig,ax=plt.subplots(figsize=(9,4))
    if kind=='band':
        keys=list(colors)
        ax.fill_between(df['Mnum'],min_y,df[keys[0]],color=colors[keys[0]],alpha=alpha)
        for i in range(1,len(keys)):
            ax.fill_between(df['Mnum'],df[keys[i-1]],df[keys[i]],color=colors[keys[i]],alpha=alpha)
        if 'Toutdoorm' in df: ax.plot(df['Mnum'],df['Toutdoorm'],'k--',linewidth=1.5)
        ax.set_ylabel('Operative Temperature (°C)')
        if max_y is not None: ax.set_ylim(min_y,max_y)
    else:
        # stacked
        cols=list(colors)
        stack=df.copy(); stack[cols]=stack[cols].div(stack[cols].sum(axis=1),axis=0)*100
        bottom=np.zeros(len(stack))
        for col in cols:
            ax.bar(stack['Mnum'],stack[col],bottom=bottom,color=colors[col],alpha=alpha)
            bottom+=stack[col].values
        ax.set_ylabel('% of Hours'); ax.set_ylim(0,100)
    ax.set_xticks(range(1,13)); ax.set_xticklabels(list(MONTH_MAP))
    ax.set_xlabel('Month'); 
    ax.grid('--',alpha=0.5)
    buf=BytesIO(); fig.savefig(buf,format='png',bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return buf

def _empty_png(w,h):
    buf=BytesIO(); fig,ax=plt.subplots(figsize=(w,h)); ax.text(0.5,0.5,'No data',ha='center'); ax.axis('off'); fig.savefig(buf,format='png',bbox_inches='tight'); buf.seek(0); plt.close(fig); return buf

# ── Shared legend endpoint ──
@app.route('/chart/legend')
@cache.cached(timeout=3600)
def chart_legend():
    colors=[
        ('Too Cold (< 80% Acceptable)','#007f9c'),
        ('Cold (80-90% Acceptable)','#0096c7'),
        ('Comfortable (90% Acceptable)','#b0b0b0'),
        ('Warm (80-90% Acceptable)','#ff7f71'),
        ('Too Hot (< 80% Acceptable)','#e63946')
    ]
    fig,ax=plt.subplots(figsize=(9,1))
    
    patches = [mpatches.Patch(color=c, label=l) for l, c in colors]
    line = Line2D([0], [0], linestyle='--', color='black', label='30 Day Running Mean')

    
    handles= patches + [line]
    ax.legend(handles=handles,loc='center',ncol=len(handles),frameon=False,fontsize='small')
    ax.axis('off')
    buf=BytesIO(); fig.savefig(buf,format='png',bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return send_file(buf,mimetype='image/png')


@app.route('/chart/combined_comfort')
def combined_comfort_chart():
    chart1_path = request.args.get('chart1')
    chart2_path = request.args.get('chart2')

    if not chart1_path or not chart2_path:
        return Response("Missing chart path(s)", status=400)

    # Load chart images and legend
    chart1 = Image.open(os.path.join(IMAGE_DIR, chart1_path))
    chart2 = Image.open(os.path.join(IMAGE_DIR, chart2_path))

    # Request legend image from Flask route
    import requests
    legend_response = requests.get(request.url_root + 'chart/legend')
    legend = Image.open(BytesIO(legend_response.content))

    # Combine vertically
    widths = [chart1.width, chart2.width, legend.width]
    heights = [chart1.height, chart2.height, legend.height]

    max_width = max(widths)
    total_height = sum(heights)

    combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in [chart1, chart2, legend]:
        combined.paste(img, (0, y_offset))
        y_offset += img.height

    # Return combined image
    buf = BytesIO()
    combined.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')




def plot_comfort_hours(df, city1, city2):
    def find_col(city, kw):
        for col in df.columns:
            if city.lower() in col.lower() and kw.lower() in col.lower():
                return col
        return None
    c1_tot = find_col(city1, 'total')
    c1_comf = find_col(city1, 'comfortable')
    c2_tot = find_col(city2, 'total')
    c2_comf = find_col(city2, 'comfortable')
    if not all([c1_tot, c1_comf, c2_tot, c2_comf]):
        buf = BytesIO()
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, 'Data missing for hours', ha='center')
        ax.axis('off')
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
    df2 = df.copy()
    df2['Temperature'] = pd.to_numeric(df2['Temperature'], errors='coerce')
    for col in [c1_tot, c1_comf, c2_tot, c2_comf]:
        df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(0)
    df2 = df2.sort_values('Temperature')
    mask = (df2[c1_tot] > 0) | (df2[c2_tot] > 0)
    sub = df2[mask] if mask.any() else df2
    tmin, tmax = sub['Temperature'].min(), sub['Temperature'].max()
    pad = 0.05 * (tmax - tmin) if tmax > tmin else 1
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(sub['Temperature'], sub[c1_comf], alpha=0.5, label=f"{city1} – Comfortable")
    ax.fill_between(sub['Temperature'], sub[c1_tot], alpha=0.3, label=f"{city1} – Total")
    ax.fill_between(sub['Temperature'], sub[c2_comf], alpha=0.5, label=f"{city2} – Comfortable")
    ax.fill_between(sub['Temperature'], sub[c2_tot], alpha=0.3, label=f"{city2} – Total")
    ax.set_xlim(tmin - pad, tmax + pad)
    ax.set_xticks(np.arange(np.floor(tmin), np.ceil(tmax)+1, 2))
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Operating Hours')
    ax.grid('--', alpha=0.5)
    
    ax.legend(
    fontsize='small',
    loc='lower center',
    bbox_to_anchor=(0.5, -0.5),ncols=4,
    frameon=False
)

    
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
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
    state = request.args.get('state', '')
    return jsonify({'cities': STATE_CITY_MAP.get(state, [])})

@app.route('/map_preview')
@cache.cached(timeout=3600, query_string=True)
def map_preview():
    c1 = request.args.get('city1', '')
    c2 = request.args.get('city2', '')
    img = composite_multi([c1, c2], [(229, 63, 70), (0, 127, 156)])
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/analyze', methods=['POST'])
def analyze():
    typ = request.form['typology']
    model = request.form['model']
    c1 = request.form['city1']
    c2 = request.form['city2']
    dfm = load_sheet('MeanMaxMin', model)
    def mk(city):
        sub = dfm[dfm['City'].str.lower() == city.lower()]
        sub = sub[['Particular'] + list(MONTH_MAP)].copy()
        sub.columns = ['Particular/Month'] + list(MONTH_MAP)
        sub[list(MONTH_MAP)] = sub[list(MONTH_MAP)].round(2)
        return sub.to_html(index=False, classes='table table-striped', border=0, float_format="%.2f")
    return jsonify(table1=mk(c1), table2=mk(c2))

@app.route('/chart/comfort1')
@cache.cached(timeout=3600, query_string=True)
def chart1():
    city  = request.args.get('city1', '')
    model = request.args.get('model', '')

    # allow override via query string
    mn = float(request.args.get('min_y', get_min_from_excel(city, model) or 0))
    mx = float(request.args.get('max_y', get_max_from_excel(city, model) or mn + 1))

    df  = load_sheet('RawData', model)
    sub = df[df['City'].str.lower() == city.lower()]

    buf = plot_comfort_bands(
        sub,
        kind='band',
        min_y=mn,
        max_y=mx
    )
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/chart/comfort2')
@cache.cached(timeout=3600, query_string=True)
def chart2():
    city  = request.args.get('city2', '')
    model = request.args.get('model', '')

    # allow override via query string
    mn = float(request.args.get('min_y', get_min_from_excel(city, model) or 0))
    mx = float(request.args.get('max_y', get_max_from_excel(city, model) or mn + 1))

    df  = load_sheet('RawData', model)
    sub = df[df['City'].str.lower() == city.lower()]

    buf = plot_comfort_bands(
        sub,
        kind='band',
        min_y=mn,
        max_y=mx
    )
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/api/minmax')
def api_minmax():
    city  = request.args['city']
    model = request.args['model']
    mn = get_min_from_excel(city, model) or 0
    mx = get_max_from_excel(city, model) or mn + 1
    return jsonify({'min_y': mn, 'max_y': mx})



@app.route('/chart/comfort_hours')
@cache.cached(timeout=3600, query_string=True)
def chart_hours():
    c1 = request.args.get('city1', '')
    c2 = request.args.get('city2', '')
    model = request.args.get('model', '')
    df = load_sheet('RawData8760', model)
    buf = plot_comfort_hours(df, c1, c2)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/download/stats1')
def download_stats1():
    city1 = request.args.get('city1', '')
    model = request.args.get('model', '')
    dfm = load_sheet('MeanMaxMin', model)
    sub = dfm[dfm['City'].str.lower() == city1.lower()]
    sub = sub[['Particular'] + list(MONTH_MAP)].copy()
    sub.columns = ['Particular/Month'] + list(MONTH_MAP)
    csv = sub.to_csv(index=False)
    return Response(csv, mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename={city1}_stats.csv'})

@app.route('/download/stats2')
def download_stats2():
    city2 = request.args.get('city2', '')
    model = request.args.get('model', '')
    dfm = load_sheet('MeanMaxMin', model)
    sub = dfm[dfm['City'].str.lower() == city2.lower()]
    sub = sub[['Particular'] + list(MONTH_MAP)].copy()
    sub.columns = ['Particular/Month'] + list(MONTH_MAP)
    csv = sub.to_csv(index=False)
    return Response(csv, mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename={city2}_stats.csv'})

# @app.route('/download/chart1')
# def download_chart1():
#     city = request.args.get('city1', '')
#     model = request.args.get('model', '')
#     df = load_sheet('RawData', model)
#     sub = df[df['City'].str.lower() == city.lower()]
#     buf = plot_comfort_bands(sub, kind='band', min_y=get_min_from_excel(city, model), max_y=get_max_from_excel(city, model))
#     buf.seek(0)
#     return send_file(buf, mimetype='image/png', as_attachment=True, attachment_filename=f'{city}_comfort_bands.png')

# @app.route('/download/chart2')
# def download_chart2():
#     city = request.args.get('city2', '')
#     model = request.args.get('model', '')
#     df = load_sheet('RawData', model)
#     sub = df[df['City'].str.lower() == city.lower()]
#     buf = plot_comfort_bands(sub, kind='band', min_y=get_min_from_excel(city, model), max_y=get_max_from_excel(city, model))
#     buf.seek(0)
#     return send_file(buf, mimetype='image/png', as_attachment=True, download_name=f'{city}_comfort_bands.png')


@app.route('/download/combined_chart')
def download_combined_chart():
    import requests
    from PIL import Image, ImageDraw, ImageFont

    city1 = request.args.get('city1', '')
    city2 = request.args.get('city2', '')
    model = request.args.get('model', '')

    if not all([city1, city2, model]):
        return Response("Missing required parameters.", status=400)

    # Construct internal URLs
    chart1_url = f"{request.url_root}chart/comfort1?city1={city1}&model={model}"
    chart2_url = f"{request.url_root}chart/comfort2?city2={city2}&model={model}"
    legend_url = f"{request.url_root}chart/legend"

    try:
        chart1_img = Image.open(BytesIO(requests.get(chart1_url).content))
        chart2_img = Image.open(BytesIO(requests.get(chart2_url).content))
        legend_img = Image.open(BytesIO(requests.get(legend_url).content))
    except Exception as e:
        return Response(f"Error fetching chart images: {str(e)}", status=500)

    # Add city name labels
    def add_city_label(img, city_name):
        font = ImageFont.load_default()
        label_height = 30
        label_img = Image.new('RGB', (img.width, label_height), (255, 255, 255))
        draw = ImageDraw.Draw(label_img)
        
        # REPLACED textsize with textbbox for compatibility
        bbox = draw.textbbox((0, 0), city_name, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (img.width - text_width) // 2

        draw.text((text_x, 10), city_name, fill=(0, 0, 0), font=font)

        combined = Image.new('RGB', (img.width, img.height + label_height), (255, 255, 255))
        combined.paste(label_img, (0, 0))
        combined.paste(img, (0, label_height))
        return combined

    labeled1 = add_city_label(chart1_img, city1)
    labeled2 = add_city_label(chart2_img, city2)

    # Combine horizontally
    total_width = labeled1.width + labeled2.width
    combined_height = max(labeled1.height, labeled2.height)
    combined = Image.new('RGB', (total_width, combined_height), (255, 255, 255))
    combined.paste(labeled1, (0, 0))
    combined.paste(labeled2, (labeled1.width, 0))

    # Add legend below
    final_width = max(combined.width, legend_img.width)
    final_height = combined.height + legend_img.height
    final_img = Image.new('RGB', (final_width, final_height), (255, 255, 255))
    final_img.paste(combined, (0, 0))
    final_img.paste(legend_img, (0, combined.height))

    # Save to buffer and send
    buf = BytesIO()
    final_img.save(buf, format='PNG')
    buf.seek(0)
    filename = f"{city1}_{city2}_comfort_chart.png"
    return send_file(buf, mimetype='image/png', as_attachment=True, download_name=filename)



@app.route('/download/chart_hours')
def download_chart_hours():
    c1 = request.args.get('city1', '')
    c2 = request.args.get('city2', '')
    model = request.args.get('model', '')
    df = load_sheet('RawData8760', model)
    buf = plot_comfort_hours(df, c1, c2)
    buf.seek(0)
    return send_file(buf, mimetype='image/png', as_attachment=True, download_name=f'{c1}_{c2}_comfort_hours.png')

if __name__=='__main__':
    app.run(debug=True, use_reloader=False)