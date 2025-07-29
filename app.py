# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import io
import base64
from datetime import datetime

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Crop Yield Prediction Dashboard"
server = app.server

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
def generate_synthetic_data(n_samples=5000):
    data = {
        'temperature': np.random.uniform(10, 40, n_samples),
        'rainfall': np.random.uniform(200, 1500, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'N': np.random.uniform(0, 200, n_samples),
        'P': np.random.uniform(0, 150, n_samples),
        'K': np.random.uniform(0, 300, n_samples),
        'ph': np.random.uniform(3.5, 9.5, n_samples),
        'organic_content': np.random.uniform(0.5, 5.0, n_samples),
        'sunlight_hours': np.random.uniform(4, 14, n_samples),
        'crop_type': np.random.choice(['wheat', 'rice', 'corn', 'soybean', 'barley'], n_samples)
    }

    df = pd.DataFrame(data)

    # Calculate Soil Health Index
    df['soil_health_index'] = (df['N']*0.3 + df['P']*0.2 + df['K']*0.2 +
                              (df['organic_content']*20) +
                              (10 - abs(df['ph'] - 6.5)*5)) / 10

    # Calculate yield based on factors with some noise
    df['yield'] = (
        (df['temperature'] * 0.5) +
        (df['rainfall'] * 0.3) +
        (df['soil_health_index'] * 50) +
        (df['sunlight_hours'] * 2) +
        np.random.normal(0, 20, n_samples)
    )

    # Clip yield to positive values
    df['yield'] = df['yield'].clip(lower=0)

    return df

# Generate and prepare the data
df = generate_synthetic_data()

# Convert categorical crop type to one-hot encoding
df_processed = pd.get_dummies(df, columns=['crop_type'], prefix='crop')

# Split into features and target
X = df_processed.drop('yield', axis=1)
y = df_processed['yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['temperature', 'rainfall', 'humidity', 'N', 'P', 'K', 'ph',
            'organic_content', 'sunlight_hours', 'soil_health_index']

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Initialize and train Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Save model and scaler

# Prediction function
def predict_yield(input_data):
    global rf, scaler
    model = rf

    input_df = pd.DataFrame([input_data])

    # Calculate soil health index
    input_df['soil_health_index'] = (
        input_df['N']*0.3 + input_df['P']*0.2 + input_df['K']*0.2 +
        (input_df['organic_content']*20) +
        (10 - abs(input_df['ph'] - 6.5)*5)
    ) / 10

    # One-hot encode crop type
    input_df = pd.get_dummies(input_df, columns=['crop_type'], prefix='crop')

    # Ensure all expected columns are present
    expected_cols = X_train.columns
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[expected_cols]

    # Scale numerical features
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Make prediction
    prediction = model.predict(input_df)

    return prediction[0]

# Optimization suggestions function
def get_yield_optimization_suggestions(input_data):
    current_yield = predict_yield(input_data)

    suggestions = {}

    # Temperature analysis
    if input_data['temperature'] < 15:
        suggestions['temperature'] = "Temperature is low for most crops. Consider using greenhouses or selecting cold-resistant varieties."
    elif input_data['temperature'] > 35:
        suggestions['temperature'] = "Temperature is high. Consider shade nets or adjusting planting dates to cooler periods."

    # Rainfall analysis
    if input_data['rainfall'] < 400:
        suggestions['rainfall'] = "Rainfall is low. Irrigation is recommended for optimal yields."
    elif input_data['rainfall'] > 1200:
        suggestions['rainfall'] = "Excessive rainfall may cause waterlogging. Ensure proper drainage."

    # Soil health analysis
    soil_health = (input_data['N']*0.3 + input_data['P']*0.2 + input_data['K']*0.2 +
                  (input_data['organic_content']*20) +
                  (10 - abs(input_data['ph'] - 6.5)*5)) / 10

    if soil_health < 5:
        suggestions['soil'] = "Soil health is poor. Consider adding organic matter and balanced fertilizers."
    elif soil_health > 8:
        suggestions['soil'] = "Soil health is excellent. Maintain current practices."

    # PH analysis
    if input_data['ph'] < 5.5:
        suggestions['ph'] = "Soil is too acidic. Consider adding lime to raise pH."
    elif input_data['ph'] > 7.5:
        suggestions['ph'] = "Soil is too alkaline. Consider adding sulfur or organic matter to lower pH."

    # Create potential improved scenarios
    improved_scenarios = []

    # Scenario 1: Optimal NPK
    scenario1 = input_data.copy()
    scenario1.update({'N': 150, 'P': 100, 'K': 200})
    improved_yield1 = predict_yield(scenario1)
    improved_scenarios.append(('Optimal NPK levels', improved_yield1))

    # Scenario 2: Optimal pH
    scenario2 = input_data.copy()
    scenario2['ph'] = 6.5
    improved_yield2 = predict_yield(scenario2)
    improved_scenarios.append(('Optimal pH (6.5)', improved_yield2))

    # Scenario 3: Increased organic content
    scenario3 = input_data.copy()
    scenario3['organic_content'] = min(4.0, scenario3['organic_content'] + 1.5)
    improved_yield3 = predict_yield(scenario3)
    improved_scenarios.append(('Increased organic content', improved_yield3))

    # Find best scenario
    best_scenario = max(improved_scenarios, key=lambda x: x[1])

    suggestions['potential_improvement'] = {
        'current_yield': current_yield,
        'best_scenario': best_scenario,
        'all_scenarios': improved_scenarios
    }

    return suggestions

# Create visualizations
def create_correlation_matrix():
    corr_matrix = df.drop('crop_type', axis=1).corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}"
    ))
    fig.update_layout(
        title='Feature Correlation Matrix',
        width=800,
        height=700
    )
    return fig

def create_yield_distribution():
    fig = px.histogram(df, x='yield', nbins=50, marginal='box',
                       title='Yield Distribution', labels={'yield': 'Yield (kg/ha)'})
    fig.update_layout(bargap=0.1)
    return fig

def create_pairplot():
    fig = px.scatter_matrix(df[['temperature', 'rainfall', 'soil_health_index', 'yield']],
                           dimensions=['temperature', 'rainfall', 'soil_health_index', 'yield'],
                           color='yield', height=800,
                           title='Pairplot of Selected Features')
    fig.update_traces(diagonal_visible=False)
    return fig

def create_feature_importance():
    fig = px.bar(feature_importance, 
                 x='Importance', 
                 y='Feature', 
                 orientation='h',
                 title='Feature Importance',
                 text=np.round(feature_importance['Importance'], 3))
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        height=500,  # Fixed height
        margin=dict(l=100, r=50, b=50, t=50),  # Adjust margins
        autosize=False  # Disable auto-sizing
    )
    
    # Adjust bar width and spacing
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        width=0.8  # Bar width
    )
    
    return fig

# Define the content for each tab
data_exploration_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Sample Data"),
            html.Div(
                dash.dash_table.DataTable(
                    id='sample-data-table',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.head(10).to_dict('records'),
                ),
                style={'height': '300px', 'overflowY': 'auto'}
            )
        ], width=6),

        dbc.Col([
            html.H3("Data Statistics"),
            html.Div(
                dash.dash_table.DataTable(
                    id='data-stats-table',
                    columns=[{"name": i, "id": i} for i in df.describe().reset_index().columns],
                    data=df.describe().reset_index().to_dict('records'),
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': 'lightgrey'},
                ),
                style={'height': '300px', 'overflowY': 'auto'}
            )
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='correlation-matrix', figure=create_correlation_matrix())
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='yield-distribution', figure=create_yield_distribution())
        ], width=6),

        dbc.Col([
            dcc.Graph(id='pairplot', figure=create_pairplot())
        ], width=6)
    ])
])

model_training_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Model Performance Metrics"),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                html.Tbody([
                    html.Tr([html.Td("Mean Absolute Error (MAE)"), html.Td(f"{mae:.2f} kg/ha")]),
                    html.Tr([html.Td("Mean Squared Error (MSE)"), html.Td(f"{mse:.2f} (kg/ha)^2")]),
                    html.Tr([html.Td("Root Mean Squared Error (RMSE)"), html.Td(f"{rmse:.2f} kg/ha")]),
                    html.Tr([html.Td("R2 Score"), html.Td(f"{r2:.4f}")]),
                ])
            ], bordered=True, striped=True)
        ], width=6),

        dbc.Col([
            html.H3("Feature Importance"),
            dcc.Graph(id='feature-importance', figure=create_feature_importance())
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H3("Preprocessed Training Data (First 10 Rows)"),
            html.Div(
                dash.dash_table.DataTable(
                    id='preprocessed-data-table',
                    columns=[{"name": i, "id": i} for i in X_train.columns],
                    data=X_train.head(10).to_dict('records'),
                ),
                style={'height': '300px', 'overflowY': 'auto'}
            )
        ], width=12)
    ])
])

yield_prediction_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Input Parameters"),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Temperature (°C)"),
                            dcc.Slider(id='temperature-slider', min=10, max=40, value=25, step=0.5,
                                      marks={i: str(i) for i in range(10, 41, 5)}),

                            dbc.Label("Rainfall (mm)"),
                            dcc.Slider(id='rainfall-slider', min=200, max=1500, value=800, step=10,
                                      marks={i: str(i) for i in range(200, 1501, 200)}),

                            dbc.Label("Humidity (%)"),
                            dcc.Slider(id='humidity-slider', min=30, max=90, value=65, step=1,
                                      marks={i: str(i) for i in range(30, 91, 10)}),

                            dbc.Label("Nitrogen (N kg/ha)"),
                            dcc.Slider(id='n-slider', min=0, max=200, value=120, step=1,
                                      marks={i: str(i) for i in range(0, 201, 25)}),

                            dbc.Label("Phosphorus (P kg/ha)"),
                            dcc.Slider(id='p-slider', min=0, max=150, value=80, step=1,
                                      marks={i: str(i) for i in range(0, 151, 25)}),
                        ], width=6),

                        dbc.Col([
                            dbc.Label("Potassium (K kg/ha)"),
                            dcc.Slider(id='k-slider', min=0, max=300, value=150, step=1,
                                      marks={i: str(i) for i in range(0, 301, 50)}),

                            dbc.Label("Soil pH"),
                            dcc.Slider(id='ph-slider', min=3.5, max=9.5, value=6.8, step=0.1,
                                      marks={3.5: '3.5', 5: '5', 6.5: '6.5', 8: '8', 9.5: '9.5'}),

                            dbc.Label("Organic Content (%)"),
                            dcc.Slider(id='organic-slider', min=0.5, max=5.0, value=2.5, step=0.1,
                                      marks={0.5: '0.5', 2.0: '2.0', 3.5: '3.5', 5.0: '5.0'}),

                            dbc.Label("Sunlight Hours"),
                            dcc.Slider(id='sunlight-slider', min=4, max=14, value=10, step=0.5,
                                      marks={i: str(i) for i in range(4, 15, 2)}),

                            dbc.Label("Crop Type"),
                            dcc.Dropdown(
                                id='crop-dropdown',
                                options=[
                                    {'label': 'Wheat', 'value': 'wheat'},
                                    {'label': 'Rice', 'value': 'rice'},
                                    {'label': 'Corn', 'value': 'corn'},
                                    {'label': 'Soybean', 'value': 'soybean'},
                                    {'label': 'Barley', 'value': 'barley'}
                                ],
                                value='wheat',
                                clearable=False
                            ),
                        ], width=6)
                    ]),

                    dbc.Button("Predict Yield", id='predict-button', color="primary", className="mt-3")
                ])
            ])
        ], width=6),

        dbc.Col([
            html.H3("Prediction & Recommendations"),
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='soil-health-output', className="mb-3"),
                    html.Div(id='yield-output', className="mb-3"),
                    html.Div(id='recommendations-output', className="mb-3"),
                    html.Div(id='improvement-output')
                ])
            ], style={'height': '100%'})
        ], width=6)
    ])
])

# Update the app.layout section (replace the existing one)
app.layout = dbc.Container([
    # Header Row
    dbc.Row([
        dbc.Col(html.H1("🌾 Crop Yield Prediction Dashboard", className="text-center my-4"), width=12)
    ]),
    
    # Main Content Tabs
    dbc.Tabs([
        dbc.Tab(label="Data Exploration", children=data_exploration_tab, tab_id="data-exploration"),
        dbc.Tab(label="Model Training", children=model_training_tab, tab_id="model-training"),
        dbc.Tab(label="Yield Prediction", children=yield_prediction_tab, tab_id="yield-prediction"),
    ], id="tabs", active_tab="data-exploration"),  # Set data-exploration as default
    
    # Footer Row
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P("© 2025 All Rights Reserved", className="text-center"),
                html.P("Made with ❤️ by Code Valley", className="text-center")
            ], className="mt-4 p-3 bg-light text-dark")
        ], width=12)
    ])
], fluid=True, style={"minHeight": "100vh", "display": "flex", "flexDirection": "column"})

# Add this CSS to make footer stick to bottom


# Add custom CSS for footer


# Callback for prediction and recommendations
@app.callback(
    [Output('soil-health-output', 'children'),
     Output('yield-output', 'children'),
     Output('recommendations-output', 'children'),
     Output('improvement-output', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('temperature-slider', 'value'),
     State('rainfall-slider', 'value'),
     State('humidity-slider', 'value'),
     State('n-slider', 'value'),
     State('p-slider', 'value'),
     State('k-slider', 'value'),
     State('ph-slider', 'value'),
     State('organic-slider', 'value'),
     State('sunlight-slider', 'value'),
     State('crop-dropdown', 'value')]
)
def update_predictions(n_clicks, temp, rain, humid, n, p, k, ph, organic, sunlight, crop):
    if n_clicks is None:
        return "", "", "", ""

    input_data = {
        'temperature': temp,
        'rainfall': rain,
        'humidity': humid,
        'N': n,
        'P': p,
        'K': k,
        'ph': ph,
        'organic_content': organic,
        'sunlight_hours': sunlight,
        'crop_type': crop
    }

    # Calculate soil health index
    shi = (n * 0.3 + p * 0.2 + k * 0.2 + (organic * 20) + (10 - abs(ph - 6.5) * 5)) / 10
    soil_health_output = dbc.Alert(f"Soil Health Index: {shi:.1f}/10", color="info")

    # Predict yield
    predicted_yield = predict_yield(input_data)
    yield_output = dbc.Alert(f"Predicted Yield: {predicted_yield:.2f} kg/ha", color="success")

    # Get recommendations
    suggestions = get_yield_optimization_suggestions(input_data)

    # Format recommendations
    recommendations = []
    for key, suggestion in suggestions.items():
        if key != 'potential_improvement':
            recommendations.append(html.P(suggestion))

    recommendations_output = dbc.Card([
        dbc.CardHeader("Recommendations"),
        dbc.CardBody(recommendations)
    ])

    # Format potential improvements
    if 'potential_improvement' in suggestions:
        imp_sugg = suggestions['potential_improvement']
        improvement_content = [
            html.H5("Potential Yield Improvement"),
            html.P(f"Current Yield: {imp_sugg['current_yield']:.2f} kg/ha"),
            html.P(f"Best Scenario ({imp_sugg['best_scenario'][0]}): {imp_sugg['best_scenario'][1]:.2f} kg/ha"),
            html.P(f"Potential Improvement: {imp_sugg['best_scenario'][1] - imp_sugg['current_yield']:.2f} kg/ha")
        ]
    else:
        improvement_content = [html.P("Could not generate potential improvement scenarios.")]

    improvement_output = dbc.Card([
        dbc.CardHeader("Potential Improvements"),
        dbc.CardBody(improvement_content)
    ])

    return soil_health_output, yield_output, recommendations_output, improvement_output

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
