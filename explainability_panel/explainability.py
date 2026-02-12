# Import packages
# import sys
# sys.path.append("deepQExpert_v2")
# sys.path.append("deepQExpert_v2/*")
import sys
sys.path.append("../.")
import warnings
warnings.filterwarnings("ignore")

from dash import Dash
import dash_bootstrap_components as dbc

from layout import layout
import callbacks.load_button
import callbacks.next_button
import callbacks.recommend_button

# import dash_ag_grid as dag
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go


external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = layout

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
