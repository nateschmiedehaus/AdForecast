import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')

def get_lat_lon(state):
    url = f'https://nominatim.openstreetmap.org/search?state={state}&format=jsonv2'
    response = requests.get(url)
    data = response.json()[0]
    return float(data['lat']), float(data['lon'])

def get_noaa_data(lat, lon):
    url = f'https://api.weather.gov/points/{lat},{lon}/forecast'
    response = requests.get(url)
    data = response.json()
    return ','.join([data['properties']['periods'][0]['temperature'], data['properties']['periods'][0]['shortForecast']])

def get_cpi_data():
    url = 'https://www.bls.gov/cpi/tables/detailed-reports/home.htm'
    tables = pd.read_html(url, match='Major')
    cpi_data = tables[0].iloc[1:, [0, 1, 2, -2, -1]]
    cpi_data.columns = ['item', 'cpi_area', 'series_id', 'jan_2021', 'feb_2021']
    return cpi_data

def get_facebook_data(api_key, api_secret):
    url = f'https://graph.facebook.com/v11.0/act_{api_key}/insights?fields=actions,spend,impressions&time_range={{%22since%22:%222023-01-01%22,%22until%22:%222023-03-24%22}}&access_token={api_key}|{api_secret}'
    response = requests.get(url)
    data = response.json()['data']
    df = pd.json_normalize(data, record_path=['actions'], meta=['spend', 'impressions'])
    df['date_start'] = pd.to_datetime(df['date_start'])
    df['cost_per_action_type'] = df['value'] / df['action_type']
    return df

def get_shopify_data(api_key, secret_key):
    url = f'https://{api_key}:{secret_key}@yourstore.myshopify.com/admin/api/2023-01/orders.json?status=any&created_at_min=2023-01-01&created_at_max=2023-03-24&fields=total_price,created_at,shipping_address'
    response = requests.get(url)
    data = response.json()['orders']
    df = pd.json_normalize(data, meta=['shipping_address'])
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    df['total_price'] = df['total_price'].astype(float)
    df['zip'] = df['shipping_address.zip'].astype(str).str[:5]
    return df


def generate_recommendations(api_key, secret_key, roas, date):
    # Retrieve data from APIs
    fb_data = get_facebook_data(api_key, date)
    shopify_data = get_shopify_data(api_key, secret_key, date)
    noaa_data = get_noaa_data_all_states(date)
    cpi_data = get_cpi_data_all_states(date)

    # Merge data into single DataFrame
    data = pd.merge(fb_data, shopify_data, on='state', how='inner')
    data = pd.merge(data, noaa_data, on='state', how='inner')
    data = pd.merge(data, cpi_data, on='state', how='inner')

    # Calculate lagged values of conversion rate and total clicks
    data['conversion_rate_lag1'] = data['conversion_rate'].shift(1)
    data['clicks_lag1'] = data['clicks'].shift(1)

    # Drop rows with missing data
    data = data.dropna()

    # Fit regression model
    X = data[['conversion_rate', 'conversion_rate_lag1', 'clicks', 'clicks_lag1']]
    y = data['total_price']
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions for each state
    states = data['state'].unique()
    recommendations = {}
    for state in states:
        # Get weather data for state
        lat, lon = get_lat_lon(state)
        forecast = get_noaa_data(lat, lon)
        temp = forecast.split(',')[0].split()[-2]
        precip = forecast.split(',')[1].split()[-2]

        # Calculate expected total price based on current and lagged values of independent variables
        row = data[data['state'] == state].iloc[-1]
        conversion_rate = row['conversion_rate']
        conversion_rate_lag1 = row['conversion_rate_lag1']
        clicks = row['clicks']
        clicks_lag1 = row['clicks_lag1']
        X_pred = np.array([[conversion_rate, conversion_rate_lag1, clicks, clicks_lag1]])
        expected_total_price = model.predict(X_pred)[0]

        # Calculate recommended spend based on expected total price and desired ROAS
        recommended_spend = (expected_total_price * roas) / fb_data[fb_data['state'] == state]['CPC'].values[0]

        # Store recommendation and weather data for state
        recommendations[state] = {
            'recommended_spend': recommended_spend,
            'expected_profit': expected_total_price,
            'temperature': temp,
            'precipitation': precip
        }

    # Write results to HTML file
    with open('results.html', 'w') as f:
        f.write('<html><head><title>WeatherAppSmall Results</title></head><body>')
        for state in states:
            # Write recommendation to table
            f.write(f'<h2>{state}</h2>')
            f.write('<table>')
            f.write('<tr><th>Recommended Spend</th><th>Expected Profit</th><th>Temperature</th><th>Precipitation</th></tr>')
            f.write(f'<tr><td>{recommendations[state]["recommended_spend"]:.2f}</td><td>{recommendations[state]["expected_profit"]:.2f}</td><td>{recommendations[state]["temperature"]}</td><td>{recommendations[state]["precipitation"]}</td></tr>')
            f.write('</table>')

            # Write recommendation to chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f'{state} - Facebook Ad Spend Recommendation')
            ax.set_xlabel('Date')
            ax.set_ylabel('Spend ($)')
            fb_data[fb_data['state'] == state].plot(x='date', y='spend', ax=ax, legend=False)
            ax.axhline(y=recommendations[state]['recommended_spend'], color='r', linestyle='--', label='Recommended Spend')
            ax.legend()
            fig.savefig(f'{state}_chart.png')

            # Write recommendation to map
            m = folium.Map(location=[39.50, -98.35], zoom_start=4)
            lat, lon = get_lat_lon(state)
            folium.Marker([lat, lon], popup=f'Recommended Spend: ${recommendations[state]["recommended_spend"]:.2f}').add_to(m)
            folium.Marker([lat, lon], icon=folium.Icon(color='red')).add_to(m)
            folium.CircleMarker([lat, lon], radius=5, color='red', fill_color='red', fill_opacity=0.5).add_to(m)
            m.save(f'{state}_map.html')
            f.write(f'<iframe src="{state}_map.html" width="100%" height="400px"></iframe>')
        f.write('</body></html>')



def generate_map(lat, lon, state_data):
    # Create map centered on state
    map_center = [lat, lon]
    m = folium.Map(location=map_center, zoom_start=6)
    # Add markers for each city in state
    for index, row in state_data.iterrows():
        marker = folium.Marker([row['latitude'], row['longitude']], tooltip=row['city'])
        marker.add_to(m)
    # Add popup showing ad spend recommendation
    popup_html = '<p>Recommended ad spend: ${:.2f}</p>'.format(state_data['Recommended Spend'][0])
    popup = folium.Popup(popup_html, max_width=200)
    folium.Marker(map_center, popup=popup).add_to(m)
    return m

def app():
    # Retrieve API keys and data
    facebook_api_key = input('Enter Facebook API key: ')
    facebook_api_secret = input('Enter Facebook API secret: ')
    shopify_api_key = input('Enter Shopify API key: ')
    shopify_secret_key = input('Enter Shopify secret key: ')
    cpi_data = get_cpi_data()
    facebook_data = get_facebook_data(facebook_api_key, facebook_api_secret)
    shopify_data = get_shopify_data(shopify_api_key, shopify_secret_key)
    # Generate recommendations for all 50 states
    roas = float(input('Enter desired ROAS: '))
    recommendations = {}
    for state in us.states.STATES:
        try:
            lat, lon = get_lat_lon(str(state))
            state_data = generate_recommendations(facebook_data, shopify_data, cpi_data, roas, lat, lon)
            recommendations[state.name] = state_data
        except:
            print(f'Error generating recommendations for {state.name}')
    # Write results to HTML file
    with open('results.html', 'w') as f:
        f.write('<html><body>')
        f.write('<h1>Ad Spend Recommendations</h1>')
        f.write('<h2>ROAS:</h2><p>{:.2f}</p>'.format(roas))
        for state, state_data in recommendations.items():
            f.write(f'<h2>{state}:</h2>')
            f.write(state_data.to_html())
            # Plot results for state
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='Conversion Rate', y='Total Price', size='Expected Profit', sizes=(50, 500), data=state_data, ax=ax)
            x_min, x_max = ax.get_xlim()
             # Plot results for state
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='Conversion Rate', y='Total Price', size='Expected Profit', sizes=(50, 500), data=state_data, ax=ax)
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            ax.plot([x_min, x_max], [state_data['model'][0].intercept, state_data['model'][0].slope * x_max + state_data['model'][0].intercept], 'r--')
            # Add weather data
            lat, lon = get_lat_lon(state)
            forecast = get_noaa_data(lat, lon)
            temp = forecast.split(',')[0].split()[-2]
            precip = forecast.split(',')[1].split()[-2]
            ax.text(x_min + 0.05 * (x_max - x_min), y_max - 0.1 * (y_max - y_min),
                    f'Temperature: {temp}\nPrecipitation: {precip}',
                    bbox=dict(facecolor='red', alpha=0.5))
            ax.set_xlabel('Conversion Rate')
            ax.set_ylabel('Total Price')
            ax.set_title(f'{state} Ad Spend Recommendations')
            fig.savefig(f'{state}.png')
            plt.close(fig)
            f.write(f'<img src="{state}.png">')
            # Generate map
            m = generate_map(lat, lon, state_data)
            m.save(f'{state}_map.html')
            f.write(f'<iframe src="{state}_map.html" width="100%" height="400px"></iframe>')
        f.write('</body></html>')

    def generate_recommendations(facebook_data, shopify_data, cpi_data, roas, lat, lon):
        # Merge data frames
        data = pd.merge(facebook_data, shopify_data, on='date')
        # Calculate conversion rate
        data['conversion_rate'] = data['purchases'] / data['clicks']
        # Fit regression model
        X = data[['conversion_rate', 'total_price']]
        y = data['total_price']
        model = LinearRegression().fit(X, y)
        # Calculate R-squared
        r_squared = model.score(X, y)
        # Calculate recommended spend
        cpi_data = cpi_data[cpi_data['cpi_area'].str.contains(state)]
        cpi = cpi_data.iloc[0]['feb_2021']
        state_returns = roas * cpi * ((model.predict(X)[-1] - model.predict(X)[0]) / X['conversion_rate'].iloc[-1])
        total_return = roas * cpi * (model.intercept_ / X['conversion_rate'].iloc[-1])
        state_recommendations = (state_returns / total_return) * 100
        recommendations_df = pd.DataFrame({'State': [state],
                                            'Conversion Rate': [X['conversion_rate'].iloc[-1]],
                                            'Total Price': [model.predict(X)[-1]],
                                            'Expected Profit': [state_returns],
                                            'Recommended Spend': [state_recommendations],
                                            'model': [model]})
        # Generate map
        state_data = pd.concat([data, recommendations_df], axis=1)
        m = generate_map(lat, lon, state_data)
        m.add_child(folium.features.GeoJson(data=us.states.MAP_SPEC, style_function=lambda x: {'color': 'gray', 'weight': 1}))
        m.add_child(folium.plugins.HeatMap(data=state_data[['latitude', 'longitude', 'Recommended Spend']], radius=15))
        m.save(f'{state}_map.html')
        # Include weather data in recommendations
        forecast = get_noaa_data(lat, lon)
        temp = forecast.split(',')[0].split()[-2]
        precip = forecast.split(',')[1].split()[-2]
        recommendations_df['Temperature'] = [temp]
        recommendations_df['Precipitation'] = [precip]
        return recommendations_df


    def get_state_recommendations(api_key, secret_key, roas, num_days):
        states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

    recommendations = {}

    # Loop over each state
    for state in states:
        print(f'Processing recommendations for {state}...')

        # Get the data for the state
        state_data = get_state_data(state)

        # Define the input variables
        X = state_data[['conversion_rate', 'total_price']]
        X_lagged = X.shift(1) # shift the data by one day to include lagged values

        # Define the dependent variable
        y = state_data['total_price']

        # Train the ARIMA model
        model = ARIMA(y, order=(1, 1, 1)) # set the order of the ARIMA model
        model_fit = model.fit()

        # Use the model to make predictions
        predictions = model_fit.predict(start=len(y), end=len(y) + num_days, dynamic=True)

        # Combine the predicted values with the lagged input variables
        predictions_df = pd.DataFrame({'total_price': predictions})
        input_df = pd.concat([X_lagged, predictions_df], axis=1)
        input_df = input_df.iloc[1:] # drop the first row because of the missing lagged values

        # Calculate the state return and total return
        state_returns = (input_df['conversion_rate'] * input_df['total_price']) / 100
        total_return = state_returns.sum()

        # Calculate the recommended spend and add it to the recommendations dictionary
        recommended_spend = roas * total_return
        recommendations[state] = {'recommended_spend': recommended_spend, 'recommendations': []}

        # Update the recommendations dictionary with the daily recommendations
        for date, row in input_df.iterrows():
            lat, lon = get_lat_lon(state)
            forecast = get_noaa_data(lat, lon)
            temp = forecast.split(',')[0].split()[-2]
            precip = forecast.split(',')[1].split()[-2]
            recommendations[state]['recommendations'].append((date.strftime('%Y-%m-%d'), row['total_price'], temp, precip))
        # Write recommendation to table
        recommendations_df = pd.DataFrame(recommendations[state]['recommendations'], columns=['Date', 'Recommended Spend', 'Temp', 'Precip'])
        recommendations_df['Temp'] = recommendations_df['Temp'].astype(int)
        recommendations_df['Precip'] = recommendations_df['Precip'].astype(int)
        recommendations_df['Spend Recommendation'] = recommended_spend * (state_returns / total_return)
        recommendations_df['Spend Recommendation'] = recommendations_df['Spend Recommendation'].astype(int)
        recommendations_df.set_index('Date', inplace=True)
        recommendations_df.index = pd.to_datetime(recommendations_df.index)
        html_table = recommendations_df.to_html()

        f.write(f'<h2>{state}</h2>')
        f.write(html_table)

        # Write recommendation to chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(recommendations_df.index, recommendations_df['Recommended Spend'], label='Recommended Spend')
        ax.plot(recommendations_df.index, recommendations_df['Spend Recommendation'], label='Spend Recommendation')
        ax.set_xlabel('Date')
        ax.set_ylabel('Spend ($)')
        ax.set_title(f'{state} Spend Recommendations')
        ax.legend()
        plt.savefig(f'{state}_chart.png')
        plt.close()

        # Write recommendation to map
        lat, lon = get_lat_lon(state)
        forecast = get_noaa_data(lat, lon)
        temp = forecast.split(',')[0].split()[-2]
        precip = forecast.split(',')[1].split()[-2]
        state_map = folium.Map(location=[lat, lon], zoom_start=6)
        HeatMap(data=recommendations_df[['Temp', 'Spend Recommendation']].values.tolist(), radius=15).add_to(state_map)
        folium.Marker(location=[lat, lon], popup=f'Temp: {temp}, Precip: {precip}').add_to(state_map)
        m.save(f'{state}_map.html')
        f.write(f'<iframe src="{state}_map.html" width="100%" height="400px"></iframe>')

    f.write('</body></html>')


def app():
    # Retrieve API keys and data
    facebook_api_key = input('Enter Facebook API key: ')
    facebook_api_secret = input('Enter Facebook API secret: ')
    shopify_api_key = input('Enter Shopify API key: ')
    shopify_secret_key = input('Enter Shopify secret key: ')
    cpi_data = get_cpi_data()
    facebook_data = get_facebook_data(facebook_api_key, facebook_api_secret)
    shopify_data = get_shopify_data(shopify_api_key, shopify_secret_key)
    # Generate recommendations for all 50 states
    roas = float(input('Enter desired ROAS: '))
    recommendations = {}
    for state in us.states.STATES:
        try:
            lat, lon = get_lat_lon(str(state))
            state_data = generate_recommendations(facebook_data, shopify_data, cpi_data, roas, lat, lon)
            recommendations[state.name] = state_data
        except:
            print(f'Error generating recommendations for {state.name}')

            # Write results to HTML file
    with open('results.html', 'w') as f:
        f.write('<html><body>')
        f.write('<h1>Ad Spend Recommendations</h1>')
        f.write('<h2>ROAS:</h2><p>{:.2f}</p>'.format(roas))
        f.write('<table><tr><th>State</th><th>Conversion Rate</th><th>Total Price</th><th>Expected Profit</th><th>Recommended Spend</th><th>Temperature</th><th>Precipitation</th></tr>')
        for state in recommendations:
            state_data = recommendations[state]
            f.write('<tr>')
            f.write(f'<td>{state}</td>')
            f.write(f'<td>{state_data["Conversion Rate"].iloc[-1]:.2f}%</td>')
            f.write(f'<td>${state_data["Total Price"].iloc[-1]:.2f}</td>')
            f.write(f'<td>${state_data["Expected Profit"].iloc[-1]:.2f}</td>')
            f.write(f'<td>{state_data["Recommended Spend"].iloc[-1]:.2f}%</td>')
            f.write(f'<td>{state_data["Temperature"].iloc[-1]}&deg; F</td>')
            f.write(f'<td>{state_data["Precipitation"].iloc[-1]}</td>')
            f.write('</tr>')
        f.write('</table>')
        f.write('<h1>Plots</h1>')
        for state in recommendations:
            f.write(f'<h2>{state}</h2>')
            f.write(f'<img src="{state}.png">')
        f.write('<h1>Maps</h1>')
        for state in recommendations:
            f.write(f'<h2>{state}</h2>')
            f.write(f'<iframe src="{state}_map.html" width="100%" height="400px"></iframe>')
        f.write('</body></html>')
