import dash
import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import preprocessing
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from dash.exceptions import PreventUpdate
import xlrd
import time



cities_list = list(pd.read_csv('city_INUS.csv')['cities'])
IN_cities_list = cities_list[0:134]
US_cities_list = cities_list[134:224]

IN_data = pd.read_excel('IN_weather2.xlsx', sheet_name = None)  # Ordered dictionary
US_data = pd.read_excel('US_weather3.xlsx', sheet_name = None)

All_coord = pd.read_csv('All_coord_IN.csv')

IN_US_result = pd.read_csv('IN_US_sim_result.csv')
US_IN_result = pd.read_csv('US_IN_sim_result.csv')

def get_coord(A):
    if any(A in s for s in cities_list):
        A_coord = eval(All_coord[All_coord['cities'].str.contains(A)]['coord'].iloc[0])
    else:
        geolocator = Nominatim(user_agent="Get coordinates")
        geolocator = Nominatim(timeout = 10)
        locationA = geolocator.geocode(A)
        A_coord = (locationA.latitude, locationA.longitude)
    return A_coord # tuple


# Find the nearest city
def find_nearest(A):
    A_coord = get_coord(A)
    distances = []
    for i in range(len(All_coord)):
        B_coord_i = eval(All_coord.iloc[i,1])
        distance_i = (A_coord[0] - B_coord_i[0])**2 + (A_coord[1] - B_coord_i[1])**2
        distances.append(distance_i)
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    nearest = All_coord.iloc[min_index, 0]
    
    actual_min = geodesic(A_coord, eval(All_coord.iloc[min_index, 1])).miles
    return nearest, actual_min

# Find the country of city A 
def findcity(cityA_input):
    if any(cityA_input in s for s in IN_cities_list):
        cityA = [s for s in IN_cities_list if cityA_input in s][0]
        countryA = 'India'
        min_distance = 0
    elif any(cityA_input in s for s in US_cities_list): 
        cityA = [s for s in US_cities_list if cityA_input in s][0]
        countryA = 'US'
        min_distance = 0
    else:
        print('Sorry your input city is not in the database but the nearest city is being investigated...')
        cityA = find_nearest(cityA_input)[0]
        min_distance = round(find_nearest(cityA_input)[1], 2)
        print('The nearest city in the database is', cityA, 'which is', min_distance, 'miles away.')
    
    if cityA in IN_cities_list:
        countryA = 'India'
    else:
        countryA = 'US'
    print(countryA)
    # ------------------------------------------------------------
    # Get all the weather data of this city
    if countryA == 'India':
        city_data = IN_data[cityA]
    else:
        city_data = US_data[cityA]
    # ------------------------------------------------------------
    #print('Your city A is', cityA, 'in', countryA)
    return city_data, countryA, min_distance, cityA


def getdata(data, data_name): 
#     average temperature = 1
#     max temp = 2
#     min temp = 3
#     precipitation = 6
#     rain days over 1mm = 7
#     days of snow = 9
#     days of storm = 10
#     days of fog = 11
#     days of frost = 12
    months = {'Months':[1,2,3,4,5,6,7,8,9,10,11,12]}
    data_item = pd.DataFrame(data = months)
    for i in range(10):
        data_item = pd.concat([data_item, data.iloc[12*i+1:12*i+13, data_name].reset_index(drop=True)], 
                            axis = 1,
                            ignore_index = True)
    data_item.columns = ['Month', '2010', '2011', '2012', '2013', '2014', '2015', '2016', 
                        '2017', '2018', '2019']
    data_item = data_item.replace(to_replace =["-"],  value = 0) 
    data_item['Mean'] = data_item.iloc[:,1:11].mean(axis = 1)
    return data_item


def find_similarity(city_data, cityB, data_name):
    if cityB in IN_cities_list: 
        country_data = IN_data
    else:
        country_data = US_data
        
    param = pd.concat([getdata(city_data, data_name)['Mean'], 
                      getdata(country_data[cityB], data_name)['Mean']], axis = 1)
    param['diffsq'] =  (param.iloc[:,0] - param.iloc[:,1]) ** 2
    similarity = param['diffsq'].sum()
    return similarity


def similarityform(city_data, countryA):
    # Construct the similarity form with all cities in another country 
    if countryA == 'India':
        citiesB_list = US_cities_list
    else: 
        citiesB_list = IN_cities_list

    similarity_form = pd.DataFrame(index = citiesB_list, 
                                   columns = ['avgtemp_sim','maxtemp_sim', 
                                              'mintemp_sim', 'prec_sim',
                                              'rain_days_sim', 'snow_days_sim',
                                              'storm_days_sim', 'fog_days_sim',
                                              'frost_days_sim'])
    for cityB in citiesB_list:
        # Find all similarities of cityB
        similarity_form.loc[cityB, 'avgtemp_sim'] = find_similarity(city_data, cityB, 1)
        similarity_form.loc[cityB, 'maxtemp_sim'] = find_similarity(city_data, cityB, 2)
        similarity_form.loc[cityB, 'mintemp_sim'] = find_similarity(city_data, cityB, 3)
        similarity_form.loc[cityB, 'prec_sim'] = find_similarity(city_data, cityB, 6)
        similarity_form.loc[cityB, 'rain_days_sim'] = find_similarity(city_data, cityB, 7)
        similarity_form.loc[cityB, 'snow_days_sim'] = find_similarity(city_data, cityB, 9)
#         similarity_form.loc[cityB, 'storm_days_sim'] = find_similarity(city_data, cityB, 10)
#         similarity_form.loc[cityB, 'fog_days_sim'] = find_similarity(city_data, cityB, 11)
#         similarity_form.loc[cityB, 'frost_days_sim'] = find_similarity(city_data, cityB, 12)
    return similarity_form


def normalize_form(similarity_form):
    # Normalize each column 
    x = similarity_form.values # Change dataframe to a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    similarity_form_normalized = pd.DataFrame(x_scaled, index = similarity_form.index, 
                                             columns = similarity_form.columns)

    # Calculate the sum of each row to get the final score 
    similarity_form_normalized['Total'] = similarity_form_normalized.sum(axis = 1)

    # Sort in ascending order
    similarity_form_normalized = similarity_form_normalized.sort_values(by = ['Total'])
    return similarity_form_normalized




rblue = '#4169e1'
mapbox_access_token = 'pk.eyJ1Ijoid2Z3aWxzb253YW5nIiwiYSI6ImNrYjQwcXJzeDBxNnUyeWxtZnlkaDF1OHoifQ.I6_PcA8uuq7kMuDh57hYJg'



app = dash.Dash()
server = app.server

app.layout = html.Div([
    # Title 
    html.H1(children = 'Cities with Similar Weather in India and the U.S.',
            style = {
                    'textAlign': 'center',
                    'color': rblue,
                    'fontsize': 20,
                    'font-family' : 'verdana'
                        }
            ),
    
    html.H3(children = 'You can find a city with the most similar weather in another country based on the weather in the past ten years.',
            style = {
                    'textAlign': 'center',
                    'font-family' : 'verdana'
                    }
           ),
    
    html.Br(),
    html.Br(),
    
    html.Div([

        dcc.Input(id = 'input-city', 
                  placeholder = 'Please enter a city name either in India or the U.S. e.g. New York, NY', 
                  type = 'text',
                  style = {'width': 500, 'height': 30, 'alignItems': 'center', 'border': 'inset', 'border-radius': '5px'}
                  ),
        
        html.Button('Find',
                    id = 'find-button',
                    style = {'width': 100, 'height': 35, 'border': 'rounded', 'border-radius': '5px'},
                    n_clicks = 0),
            ],
        style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    
    html.Div(
         html.H5(id = 'find-result-input'),
         style = {'textAlign': 'center', 'font-family' : 'verdana'} ),
    
    html.Div(
         html.H5(id = 'find-result'),
         style = {'textAlign': 'center', 'font-family' : 'verdana', 'color': rblue} ),   
    
    html.Div(
         [dcc.Graph(id = 'two-city-map',
                   style = {'height': '100%'},
                   figure = {
                            'data': [go.Scattermapbox(
                                        lat = [],
                                        lon = [],
                                        mode = 'markers',
                                        marker=go.scattermapbox.Marker(size=14))],
                            'layout': go.Layout(hovermode = 'closest',
                                        mapbox = dict(
                                            accesstoken = mapbox_access_token,
                                            bearing = 0,
                                            pitch = 0,
                                        ))
                   })],
        style = {'display': 'inline-block', 'height': 600, 'width': '100%'}),
    
    html.Div([
        dcc.Graph(id = 'avg-temp',  # fig2
                  style = {'width': '33%',  'display': 'inline-block'}),
        dcc.Graph(id = 'max-temp',  # fig3
                  style = {'width': '33%',  'display': 'inline-block'}),
        dcc.Graph(id = 'min-temp',  # fig4
                  style = {'width': '33%',  'display': 'inline-block'}),
        dcc.Graph(id = 'precipitation',  # fig5
                  style = {'width': '33%',  'display': 'inline-block'}),
        dcc.Graph(id = 'days-prec',  # fig6
                  style = {'width': '33%',  'display': 'inline-block'}),
        dcc.Graph(id = 'days-snow',  # fig7
                  style = {'width': '33%',  'display': 'inline-block'}),
#         dcc.Graph(id = 'days-storm',  # fig8
#                   style = {'width': '33%',  'display': 'inline-block'}),
#         dcc.Graph(id = 'days-fog',  # fig9
#                   style = {'width': '33%',  'display': 'inline-block'}),
#         dcc.Graph(id = 'days-frost',  # fig10
#                   style = {'width': '33%',  'display': 'inline-block'}),    
    ])
])

@app.callback(
    Output('find-result-input', 'children'),
    [Input('input-city', 'value')])
def update_output(cityA_input):
    
    return 'You have entered the city "{}"'.format(cityA_input)


@app.callback(
    [Output('find-result', 'children'),
     Output('two-city-map', 'figure'),
     Output('avg-temp', 'figure'),
     Output('max-temp', 'figure'),
     Output('min-temp', 'figure'),
     Output('precipitation', 'figure'),
     Output('days-prec', 'figure'),
     Output('days-snow', 'figure')],
#      Output('days-storm', 'figure'),
#      Output('days-fog', 'figure'),
#      Output('days-frost', 'figure')],
    [Input('find-button', 'n_clicks')],
    [State('input-city', 'value' )])
def update_output2(n_clicks, cityA_input):
    if cityA_input is None:
        raise PreventUpdate
    
    # Result city
    cityA = str(cityA_input)
    if cityA in IN_cities_list:
        cityA_data = IN_data[cityA]
        cityB_result = IN_US_result[IN_US_result['IN_cities'].str.contains(cityA)].iloc[0,1]
        result = 'City with the most similar weather with {} is {}'.format(cityA_input, cityB_result)
    elif cityA in US_cities_list:
        cityA_data = US_data[cityA]
        cityB_result = US_IN_result[US_IN_result['US_cities'].str.contains(cityA)].iloc[0,1]
        result = 'City with the most similar weather with {} is {}'.format(cityA_input, cityB_result)
    else: 
        cityA_data, countryA, min_distance, city_near = findcity(cityA)
        if city_near in IN_cities_list:
            cityB_result = IN_US_result[IN_US_result['IN_cities'].str.contains(city_near)].iloc[0,1]
        elif city_near in US_cities_list:
            cityB_result = US_IN_result[US_IN_result['US_cities'].str.contains(city_near)].iloc[0,1]

        result = 'Sorry your input city is not in the database but the nearest city is {} in {} miles away.\n \
                  City with the most similar weather with {} is {}'.format(city_near, min_distance, city_near, cityB_result)
    
    
    # Map 
    coord_A = get_coord(cityA)
    coord_B = get_coord(cityB_result)
    
    fig = go.Figure(go.Scattermapbox(
        lat = [coord_A[0], coord_B[0]],
        lon = [coord_A[1], coord_B[1]],
        mode = 'markers',
        marker=go.scattermapbox.Marker(
            size=14
        ),
    ))
    
    fig.update_layout(
    hovermode = 'closest',
    mapbox = dict(
        accesstoken = mapbox_access_token,
        bearing = 0,
        center = go.layout.mapbox.Center(
            lat = 40.52,
            lon = 34.34
        ), 
        pitch = 0,
#         zoom = 5
    )
    )
    
    # Plots 
    cityB_data = findcity(cityB_result)[0]
    
    
    # Average temperature
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityA_data, 1)['Mean'],
                    mode = 'lines+markers',
                    name = cityA))
    fig2.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityB_data, 1)['Mean'],
                    mode = 'lines+markers',
                    name = cityB_result))  
    fig2.update_layout(
        {'title': 'Average Temperature (°C) ',
         'xaxis': {'title': 'Month'},
         'yaxis': {'title': 'Temperature (°C)'}}
    )
    
    # Max temperature
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityA_data, 2)['Mean'],
                    mode = 'lines+markers',
                    name = cityA))
    fig3.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityB_data, 2)['Mean'],
                    mode = 'lines+markers',
                    name = cityB_result))  
    fig3.update_layout(
        {'title': 'Max Temperature (°C)',
         'xaxis': {'title': 'Month'},
         'yaxis': {'title': 'Temperature (°C)'}}
    )    
    
    # Min temperature
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityA_data, 3)['Mean'],
                    mode = 'lines+markers',
                    name = cityA))
    fig4.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityB_data, 3)['Mean'],
                    mode = 'lines+markers',
                    name = cityB_result))  
    fig4.update_layout(
        {'title': 'Min Temperature (°C)',
         'xaxis': {'title': 'Month'},
         'yaxis': {'title': 'Temperature (°C)'}}
    )    
    
    # Precipitation
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityA_data, 6)['Mean'],
                    mode = 'lines+markers',
                    name = cityA))
    fig5.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityB_data, 6)['Mean'],
                    mode = 'lines+markers',
                    name = cityB_result))  
    fig5.update_layout(
        {'title': 'Precipitation (mm)',
         'xaxis': {'title': 'Month'},
         'yaxis': {'title': 'Precipitation (mm)'}}
    )     
    
    # Days of precipitation
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityA_data, 7)['Mean'],
                    mode = 'lines+markers',
                    name = cityA))
    fig6.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityB_data, 7)['Mean'],
                    mode = 'lines+markers',
                    name = cityB_result))  
    fig6.update_layout(
        {'title': 'Days of precipitation > 1mm',
         'xaxis': {'title': 'Month'},
         'yaxis': {'title': 'Days'}}
    )        
    
   
    # Days of snow
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityA_data, 9)['Mean'],
                    mode = 'lines+markers',
                    name = cityA))
    fig7.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityB_data, 9)['Mean'],
                    mode = 'lines+markers',
                    name = cityB_result))  
    fig7.update_layout(
        {'title': 'Days of snow',
         'xaxis': {'title': 'Month'},
         'yaxis': {'title': 'Days'}}
    )        
    
#     # Days of storm
#     fig8 = go.Figure()
#     fig8.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityA_data, 10)['Mean'],
#                     mode = 'lines+markers',
#                     name = cityA))
#     fig8.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityB_data, 10)['Mean'],
#                     mode = 'lines+markers',
#                     name = cityB_result))  
#     fig8.update_layout(
#         {'title': 'Days of storm',
#          'xaxis': {'title': 'Month'},
#          'yaxis': {'title': 'Days'}}
#     )      
    
#     # Days of fog
#     fig9 = go.Figure()
#     fig9.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityA_data, 11)['Mean'],
#                     mode = 'lines+markers',
#                     name = cityA))
#     fig9.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityB_data, 11)['Mean'],
#                     mode = 'lines+markers',
#                     name = cityB_result))  
#     fig9.update_layout(
#         {'title': 'Days of fog',
#          'xaxis': {'title': 'Month'},
#          'yaxis': {'title': 'Days'}}
#     )      
    
#     # Days of frost
#     fig10 = go.Figure()
#     fig10.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityA_data, 12)['Mean'],
#                     mode = 'lines+markers',
#                     name = cityA))
#     fig10.add_trace(go.Scatter(x = np.arange(1, 13), y = getdata(cityB_data, 12)['Mean'],
#                     mode = 'lines+markers',
#                     name = cityB_result))  
#     fig10.update_layout(
#         {'title': 'Days of frost',
#          'xaxis': {'title': 'Month'},
#          'yaxis': {'title': 'Days'}}
#     )      
    
    return result, fig, fig2, fig3, fig4, fig5, fig6, fig7#, fig8, fig9, fig10


if __name__ == '__main__':
    app.run_server()
