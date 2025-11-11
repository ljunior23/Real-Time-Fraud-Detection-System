
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import requests
from datetime import datetime
from collections import deque

"""
Simple monitoring dashboard using Plotly Dash
Visualizes real-time fraud detection metrics
"""

class MonitoringDashboard:
    def __init__(self, api_url='http://localhost:8000', update_interval=2000):
        self.api_url = api_url
        self.update_interval = update_interval
        
        # Store metrics history
        self.metrics_history = {
            'timestamps': deque(maxlen=100),
            'total_processed': deque(maxlen=100),
            'fraud_detected': deque(maxlen=100),
            'precision': deque(maxlen=100),
            'recall': deque(maxlen=100),
            'avg_latency': deque(maxlen=100)
        }
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Real-Time Fraud Detection Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            html.Div([
                html.Div([
                    html.H3("Transactions Processed"),
                    html.H2(id='total-processed', children='0'),
                ], className='metric-box'),
                
                html.Div([
                    html.H3("Frauds Detected"),
                    html.H2(id='fraud-detected', children='0', 
                           style={'color': '#e74c3c'}),
                ], className='metric-box'),
                
                html.Div([
                    html.H3("Precision"),
                    html.H2(id='precision', children='0.00'),
                ], className='metric-box'),
                
                html.Div([
                    html.H3("Recall"),
                    html.H2(id='recall', children='0.00'),
                ], className='metric-box'),
            ], style={'display': 'flex', 'justifyContent': 'space-around'}),
            
            html.Br(),
            
            dcc.Graph(id='throughput-graph'),
            dcc.Graph(id='fraud-rate-graph'),
            dcc.Graph(id='latency-graph'),
            dcc.Graph(id='confusion-matrix'),
            
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ])
    
    def fetch_metrics(self):
        """Fetch metrics from API"""
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=2)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('total-processed', 'children'),
             Output('fraud-detected', 'children'),
             Output('precision', 'children'),
             Output('recall', 'children'),
             Output('throughput-graph', 'figure'),
             Output('fraud-rate-graph', 'figure'),
             Output('latency-graph', 'figure'),
             Output('confusion-matrix', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            metrics = self.fetch_metrics()
            
            if metrics is None:
                # Return empty values if API not available
                return '0', '0', '0.00', '0.00', {}, {}, {}, {}
            
            # Update history
            self.metrics_history['timestamps'].append(datetime.now())
            self.metrics_history['total_processed'].append(
                metrics.get('transactions_processed', 0)
            )
            # Note: These would come from actual metrics collection
            # For now, using placeholder logic
            
            # Create figures
            throughput_fig = self.create_throughput_graph()
            fraud_rate_fig = self.create_fraud_rate_graph()
            latency_fig = self.create_latency_graph()
            cm_fig = self.create_confusion_matrix()
            
            return (
                f"{metrics.get('transactions_processed', 0):,}",
                "N/A",  # Would come from consumer metrics
                "N/A",
                "N/A",
                throughput_fig,
                fraud_rate_fig,
                latency_fig,
                cm_fig
            )
    
    def create_throughput_graph(self):
        """Create throughput over time graph"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(self.metrics_history['timestamps']),
            y=list(self.metrics_history['total_processed']),
            mode='lines',
            name='Transactions',
            line=dict(color='#3498db', width=2)
        ))
        fig.update_layout(
            title='Transaction Throughput Over Time',
            xaxis_title='Time',
            yaxis_title='Total Transactions',
            hovermode='x'
        )
        return fig
    
    def create_fraud_rate_graph(self):
        """Create fraud detection rate graph"""
        fig = go.Figure()
        # Placeholder implementation
        fig.update_layout(title='Fraud Detection Rate Over Time')
        return fig
    
    def create_latency_graph(self):
        """Create latency distribution graph"""
        fig = go.Figure()
        # Placeholder implementation
        fig.update_layout(title='Prediction Latency Distribution')
        return fig
    
    def create_confusion_matrix(self):
        """Create confusion matrix heatmap"""
        fig = go.Figure()
        # Placeholder implementation
        fig.update_layout(title='Confusion Matrix')
        return fig
    
    def run(self, host='0.0.0.0', port=8050):
        """Run dashboard"""
        print(f"✓ Dashboard starting on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=False)


if __name__ == "__main__":
    dashboard = MonitoringDashboard()
    dashboard.run()