import base64  # for encoding uploaded file content
import io      # to wrap decoded bytes for pandas
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go  # for creating empty figures on no data

# --- Global state ---
global_df = None
global_model = None
global_features = []
global_num_feats = []
global_cat_feats = []

# --- Dash app setup ---
app = dash.Dash(__name__)
server = app.server

# --- Layout ---
app.layout = html.Div([
    html.H1('Data Analysis and Prediction App'),

# 1. Upload section
html.Div([
    html.H2('Upload File', className='upload-header'),
    dcc.Upload(
        id='upload-data',
        children=html.Div('Drag and Drop or Click to Select a CSV File'),
        className='upload-box',
        multiple=False
    ),
    html.Div(id='upload-status')
], className='upload-container'),
html.Hr(),

# 2. Target selection
html.Div([
    html.Label('Select Target:', className='target-label'),
    dcc.Dropdown(
        id='target-dropdown',
        options=[],
        placeholder='Choose target variable',
        className='target-dropdown'
    )
], className='target-selector', style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),
html.Hr(),

# 3. Exploratory charts side-by-side
html.Div([
    html.Div([
        html.Div(
            dcc.RadioItems(
                id='cat-radio',
                options=[],
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            ),
            style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}
        ),
        dcc.Graph(id='avg-bar-chart', figure=go.Figure())
    ], style={'width': '48%'}),
    html.Div([
        dcc.Graph(id='corr-bar-chart', figure=go.Figure())
    ], style={'width': '48%'})
], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '10px'}),
html.Hr(),

# 4. Train model
html.Div([
    html.Div(
        dcc.Checklist(
            id='feature-checklist',
            options=[],
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        ),
        className='train-checklist-container'
    ),
    html.Div(
        html.Button('Train', id='train-button'),
        className='train-button-container'
    )
], className='train-section'),

# output below train
html.Div(
    id='train-output',
    className='train-output-container'
),
html.Hr(),

# 5. Predict
html.Div([
    html.Div([
        dcc.Input(
            id='predict-input',
            type='text',
            placeholder='Enter comma-separated feature values',
            className='predict-input'
        ),
        html.Button('Predict', id='predict-button', className='predict-button'),
        html.Div(id='predict-output', className='predict-output-inline')
    ], className='predict-controls')
], className='predict-section')
])

# --- Callback: upload file ---
@app.callback(
    Output('upload-status', 'children'),
    Output('target-dropdown', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    global global_df
    if not contents:
        return '', []
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception:
        return 'Error processing file.', []
    # map ordinal values for 'size'
    if 'size' in df.columns:
        size_map = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6}
        df['size'] = df['size'].map(size_map)
    global_df = df
    nums = df.select_dtypes(include=np.number).columns.tolist()
    options = [{'label': c, 'value': c} for c in nums]
    return f'Loaded "{filename}" with {df.shape[0]} rows and {df.shape[1]} columns', options

# --- Callback: update options ---
@app.callback(
    Output('cat-radio', 'options'),
    Output('feature-checklist', 'options'),
    Input('target-dropdown', 'value')
)
def update_options(target):
    if global_df is None or not target:
        return [], []
    df = global_df
    cat_opts = [{'label': c, 'value': c} for c in df.select_dtypes(include=['object','category']).columns]
    feat_opts = [{'label': c, 'value': c} for c in df.columns if c != target]
    return cat_opts, feat_opts

# --- Callback: average bar chart ---
@app.callback(
    Output('avg-bar-chart', 'figure'),
    Input('target-dropdown', 'value'),
    Input('cat-radio', 'value')
)
def update_avg_chart(target, cat_col):
    if global_df is None or not target or not cat_col:
        return go.Figure()
    data = global_df.groupby(cat_col)[target].mean().reset_index()
    fig = px.bar(
        data, x=cat_col, y=target,
        labels={target: f"{target} (average)"},
        text_auto='.6f', template='plotly_white'
    )
    fig.update_traces(width=0.6, marker_color='skyblue', marker_line_color='rgba(79,129,189,1.0)',
                      marker_line_width=1.5, textfont_size=12, textfont_color='rgba(50,50,50,0.8)',
                      textposition='inside')
    fig.update_layout(title={'text': f"<b>Average {target} by {cat_col}</b>", 'x': 0.5, 'xanchor': 'center'},
                      font=dict(family='Times New Roman, Times, serif', size=12),
                      yaxis=dict(range=[0, data[target].max() * 1.1]), bargap=0.4)
    return fig

# --- Callback: correlation chart ---
@app.callback(
    Output('corr-bar-chart', 'figure'),
    Input('target-dropdown', 'value')
)
def update_corr_chart(target):
    if global_df is None or not target:
        return go.Figure()
    df = global_df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    if not numeric_cols:
        return go.Figure()
    corrs = df[numeric_cols].corrwith(df[target]).abs().reset_index()
    corrs.columns = ['feature', 'correlation']
    fig = px.bar(
        corrs, x='feature', y='correlation',
        labels={'feature': 'Numerical Variables', 'correlation': 'Correlation Strength (Absolute Value)'},
        text_auto='.2f', template='plotly_white'
    )
    fig.update_traces(width=0.6, marker_line_color='rgba(79,129,189,1.0)', marker_line_width=1.5,
                      textfont_size=12, textfont_color='white', textposition='inside')
    fig.update_layout(title={'text': f"<b>Correlation Strength of Numerical Variables with {target}</b>", 'x': 0.5, 'xanchor': 'center'},
                      font=dict(family='Times New Roman, Times, serif', size=12),
                      yaxis=dict(range=[0, corrs['correlation'].max() * 1.1]), bargap=0.4)
    return fig

# --- Callback: train model ---
@app.callback(
    Output('train-output', 'children'),
    Input('train-button', 'n_clicks'),
    State('target-dropdown', 'value'),
    State('feature-checklist', 'value')
)
def train_model(n_clicks, target, features):
    global global_model, global_features, global_num_feats, global_cat_feats
    if not n_clicks or global_df is None or not target or not features:
        return ''
    # drop rows with missing target
    df_train = global_df.dropna(subset=[target])
    X = df_train[features]
    y = df_train[target]
    # identify numeric vs categorical features
    num_feats = [c for c in features if c in X.select_dtypes(include=np.number).columns]
    cat_feats = [c for c in features if c in X.select_dtypes(include=['object','category']).columns]
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('num', num_pipe, num_feats), ('cat', cat_pipe, cat_feats)])
    pipeline = Pipeline([('pre', pre), ('rf', RandomForestRegressor(n_estimators=100, random_state=42))])
    pipeline.fit(X, y)
    score = pipeline.score(X, y)
    global_model = pipeline
    global_features = features
    global_num_feats = num_feats
    global_cat_feats = cat_feats
    return f'The R² score is: {score:.3f}'

# --- Callback: predict ---
@app.callback(
    Output('predict-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('predict-input', 'value'),
    State('target-dropdown', 'value')
)
def predict(n_clicks, input_str, target):
    if not n_clicks or global_model is None or not input_str or not target:
        return ''
    raw_vals = [v.strip() for v in input_str.split(',')]
    if len(raw_vals) != len(global_features):
        return f'Expected {len(global_features)} values, got {len(raw_vals)}'
    row = {}
    for feat, val in zip(global_features, raw_vals):
        if feat in global_num_feats:
            try:
                row[feat] = float(val)
            except ValueError:
                row[feat] = np.nan
        else:
            row[feat] = val
    X_new = pd.DataFrame([row], columns=global_features)
    try:
        pred = global_model.predict(X_new)[0]
    except Exception as e:
        return f'Prediction error: {e}'
    if X_new.isna().any().any():
        missing = X_new.columns[X_new.isna().any()].tolist()
        return f'Non-numeric values in {missing!r} treated as missing and imputed.  Predicted {target} is: {pred:.3f}'
    return f'Predicted {target} is: {pred:.3f}'

# --- Run server ---
if __name__ == '__main__':
    app.run_server(debug=True)
