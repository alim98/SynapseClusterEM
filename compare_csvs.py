import numpy as np
import pandas as pd
import plotly.graph_objects as go

def create_interactive_plot(umap_data, sample_pairs, output_file, max_pairs=None):
    """Create an interactive plot with sample pair connections and a slider to control the number of displayed pairs."""
    
    # Limit the number of pairs to avoid performance issues
    max_pairs = min(max_pairs or len(sample_pairs), len(sample_pairs))
    
    # Create a figure
    fig = go.Figure()
    
    # Separate points for better visualization based on the data
    n_points = len(umap_data)
    mid_point = n_points // 2 if n_points > len(sample_pairs) * 2 else None
    
    # Add all points
    if mid_point:
        # First half of points (first dataset)
        fig.add_trace(go.Scatter(
            x=umap_data[:mid_point, 0],
            y=umap_data[:mid_point, 1],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.7),
            name='Stage-Specific Layer 20'
        ))
        
        # Second half of points (second dataset)
        fig.add_trace(go.Scatter(
            x=umap_data[mid_point:, 0],
            y=umap_data[mid_point:, 1],
            mode='markers',
            marker=dict(size=8, color='red', opacity=0.7),
            name='Standard Extraction'
        ))
    else:
        # All points in one trace
        fig.add_trace(go.Scatter(
            x=umap_data[:, 0],
            y=umap_data[:, 1],
            mode='markers',
            marker=dict(size=8, color='lightgray'),
            name='All Points'
        ))
    
    # Add a trace for each connection that will be shown/hidden by the slider
    for i in range(max_pairs):
        if i < len(sample_pairs):
            pair = sample_pairs[i]
            fig.add_trace(go.Scatter(
                x=[umap_data[pair[0], 0], umap_data[pair[1], 0]],
                y=[umap_data[pair[0], 1], umap_data[pair[1], 1]],
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.4)', width=1),
                showlegend=(i==0),  # Only show legend for the first connection
                visible=False,      # Initially hidden
                name='Connection'
            ))
    
    # Create the slider
    steps = []
    for i in range(max_pairs + 1):
        # For each step, determine which traces should be visible
        # Scatter points are always visible (first 1 or 2 traces)
        visible_list = [True, True] if mid_point else [True]
        
        # Add visibility for each connection line
        for j in range(max_pairs):
            visible_list.append(j < i)  # Show connections up to current slider value
        
        step = dict(
            method="update",
            args=[{"visible": visible_list}],
            label=f"{i}"
        )
        steps.append(step)
    
    # Add the slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Number of connections: ", "font": {"size": 14, "color": "#000000"}},
        pad={"t": 50, "b": 10},
        len=0.9,
        x=0.1,
        xanchor="left",
        y=-0.1,     # Position slider below the plot
        yanchor="top",
        steps=steps,
        transition={"duration": 0},
        bgcolor="#f0f0f0",  # Light gray background for the slider
        bordercolor="#dbdbdb",
        borderwidth=2,
        ticklen=5
    )]
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive UMAP Comparison: Stage-Specific vs Standard Extraction',
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        width=1000,
        height=700,
        template='plotly_white',
        sliders=sliders,
        margin=dict(l=50, r=50, t=100, b=150),  # Increased bottom margin for slider
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.15,  # Position below the plot
                text="Use slider to control number of connections shown",
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )
    
    # Save to HTML with full HTML wrapper
    config = {
        'displayModeBar': True,
        'responsive': True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'umap_comparison',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }
    
    # Use a direct HTML approach to ensure one plot
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>UMAP Comparison with Slider Control</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                font-family: Arial, sans-serif;
            }}
            .plot-container {{
                width: 100%;
                max-width: 1200px;
                margin: 0 auto;
                height: 800px;
            }}
            .slider-label {{
                text-align: center;
                font-size: 16px;
                font-weight: bold;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="plot-container" id="plot"></div>
        <script>
            {fig.to_json()}
            Plotly.newPlot('plot', figure.data, figure.layout, {config});
        </script>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive visualization with slider saved to {output_file}")
    
    return fig 