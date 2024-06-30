import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances_argmin_min
from data_loader import load_data
from plotly_chart import plotly_scatter_chart
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import uuid

dataset_key = str(uuid.uuid4())
tabs_key = str(uuid.uuid4())
slider_key = str(uuid.uuid4())
slider_key2 = str(uuid.uuid4())

dataset = st.sidebar.radio("Choose Dataset", ["SLB Data", "Relocated Data"])
tabs = st.sidebar.radio("Choose Visualization", ["Line Chart", "Bar Chart", "Scatter Plot", "Custom Plotly Scatter Chart"])
k_clusters = st.sidebar.slider("Select Number of Clusters", min_value=1, max_value=20, value=8, key="kmeans_slider")

def generate_date_input_key(tab_name, date_type):
    return f"date_input_{tab_name}_{date_type}"

def get_date_input(tab_name):
    start_date_kmeans = st.date_input('Start Date K-Means', pd.to_datetime('01/01/2011', format='%d/%m/%Y'), key=generate_date_input_key(tab_name, "start"))
    end_date_kmeans = st.date_input('End Date K-Means', pd.to_datetime('31/12/2018', format='%d/%m/%Y'), key=generate_date_input_key(tab_name, "end"))
    return start_date_kmeans, end_date_kmeans

def load_slb_data_with_date_range(start_date, end_date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    slb_data = load_data("SLB Data", start_date, end_date)
    return slb_data

def load_relocated_data_with_date_range(start_date, end_date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    relocated_data = load_data("Relocated Data", start_date, end_date)
    return relocated_data

def data_visualization_page():
    st.title('Data Visualization')

    st.markdown("""
    **Note:**

    This page allows you to visualize depth difference data for the selected dataset. 

    * Select the **Start Date** and **End Date** for the visualization period.
    * Choose the desired type of chart from the available options: Line Chart, Bar Chart, Scatter Plot, or Custom Plotly Scatter Chart.
    * Pick a dataset from the options on the side.

    **Please note:** 
    * The available data columns and chart types may vary depending on the selected dataset.
    * For custom Plotly charts, refer to the documentation for additional customization options.
    """)
    
    start_date_viz = st.date_input('Start Date Visualization', pd.to_datetime('01/01/2011', format='%d/%m/%Y'))
    end_date_viz = st.date_input('End Date Visualization', pd.to_datetime('31/12/2018', format='%d/%m/%Y'))

    start_date = pd.Timestamp(start_date_viz)
    end_date = pd.Timestamp(end_date_viz)

    data = load_data(dataset, start_date, end_date)

    origin_time_col = "SLB origin time" if dataset == "SLB Data" else "Relocated origin time"
    depth_difference_col = "SLB Depth Difference" if dataset == "SLB Data" else "Relocated Depth Difference"

    if tabs == "Line Chart":
        st.line_chart(data.set_index(origin_time_col)[depth_difference_col])

    elif tabs == "Bar Chart":
        st.bar_chart(data.set_index(origin_time_col)[depth_difference_col])

    elif tabs == "Scatter Plot":
        st.scatter_chart(data.set_index(origin_time_col)[depth_difference_col])

    elif tabs == "Custom Plotly Scatter Chart":
        st.plotly_chart(plotly_scatter_chart(data, origin_time_col, depth_difference_col, 'Year/Mo. Category'))

    if st.button("Export Selected Data"):
        csv = data.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name='selected_data.csv', mime='text/csv')

def perform_kmeans_clustering(slb_data, k_clusters=8, features=['SLB Depth Difference', 'SLB Horizontal Difference']):
    st.markdown("""
    Performs K-Means clustering on the provided dataset.

    This function groups similar data points into clusters based on their features. 
    K-Means clustering is an unsupervised learning technique that aims to partition data 
    into a predefined number of clusters without any prior knowledge of the labels.

    In this context, we use K-Means clustering to identify groups of seismic line 
    segments with similar depth and horizontal difference characteristics. This can help 
    us understand the potential presence of geological formations or patterns within 
    the seismic data.
    
    **Cluster Formats:**

    The list displayed below the plot shows the unique "Year/Mo. Category" formats present within that cluster. This information can help:

    * Identify dominant formations within each cluster.
    * Investigate potential relationships between formations and cluster characteristics.
    * Guide further analysis or interpretation of the seismic data.
    """)
    st.write("Performing K-Means Clustering on Dataset...")

    slb_data['Year/Mo. Category'] = slb_data['Year/Mo. Category'].astype(str)

    data_for_clustering = slb_data[features].dropna()

    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    slb_data['cluster'] = kmeans.fit_predict(data_for_clustering)

    # Sample for silhouette score calculation to avoid MemoryError
    sample_size = min(1000, len(data_for_clustering))  # Use a smaller sample if the dataset is very large
    if len(data_for_clustering) > sample_size:
        sample_data = data_for_clustering.sample(n=sample_size, random_state=42)
        sample_labels = slb_data.loc[sample_data.index, 'cluster']
    else:
        sample_data = data_for_clustering
        sample_labels = slb_data['cluster']

    silhouette_avg = silhouette_score(sample_data, sample_labels)
    db_index = davies_bouldin_score(data_for_clustering, slb_data['cluster'])
    st.write(f'Silhouette Score for K-Means (sampled): {silhouette_avg}')
    st.write(f'Davies-Bouldin Index for K-Means: {db_index}')

    palette = sns.color_palette("hsv", 20)
    custom_colors = [sns.color_palette(palette).as_hex()[i] for i in range(20)][:k_clusters]

    fig = go.Figure()

    for cluster_id in range(k_clusters):
        cluster_data = slb_data[slb_data['cluster'] == cluster_id]
        fig.add_trace(go.Scatter(x=cluster_data['SLB Horizontal Difference'],
                                 y=cluster_data['SLB Depth Difference'],
                                 mode='markers',
                                 marker=dict(color=custom_colors[cluster_id]),
                                 name=f'Cluster {cluster_id}'))
        
    mountain_heights = {'Precambrian': -200, 'Argenta': 0, 'Mt Simon A Lower': 100, 'Mt Simon A Upper': 600, 'Mt Simon B': 800, 'Mt Simon C': 1000}
    for mountain, height in mountain_heights.items():
        fig.add_shape(type="line", x0=min(slb_data['SLB Horizontal Difference']), y0=height, x1=max(slb_data['SLB Horizontal Difference']), y1=height,
                      line=dict(color="red", width=2))
        fig.add_annotation(text=mountain, xref="x", yref="y", x=max(slb_data['SLB Horizontal Difference']), y=height, showarrow=False, yshift=5, xshift=10)

    fig.update_layout(title='K-Means Clustering of SLB Data',
                      xaxis_title='SLB Horizontal Difference',
                      yaxis_title='SLB Depth Difference')

    st.plotly_chart(fig)
    
    cluster_formats = slb_data.groupby('cluster')['Year/Mo. Category'].unique()
    for cluster_id, formats_in_cluster in cluster_formats.items():
        st.write(f"Formats in Cluster {cluster_id}:", formats_in_cluster)

def kmeans_clustering_page(start_date, end_date, tab_name, k_clusters):
    st.title('K-Means Clustering')

    slb_data_for_clustering = load_slb_data_with_date_range(start_date, end_date)
    
    perform_kmeans_clustering(slb_data_for_clustering, k_clusters=k_clusters)

    return slb_data_for_clustering

def kmeans_3d_clustering_page(start_date, end_date, k_clusters):
    st.title('3D K-Means Clustering')

    slb_data = load_slb_data_with_date_range(start_date, end_date)
    slb_data['SLB origin time'] = pd.to_datetime(slb_data['SLB origin time'])
    slb_data['TimeInSeconds'] = (slb_data['SLB origin time'] - slb_data['SLB origin time'].min()).dt.total_seconds()
    features = ['TimeInSeconds', 'SLB Depth Difference', 'SLB Horizontal Difference']

    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    slb_data['Cluster'] = kmeans.fit_predict(slb_data[features])

    fig = go.Figure()

    for cluster_id in range(k_clusters):
        cluster_data = slb_data[slb_data['Cluster'] == cluster_id]
        fig.add_trace(go.Scatter3d(x=cluster_data['TimeInSeconds'], 
                                   y=cluster_data['SLB Depth Difference'], 
                                   z=cluster_data['SLB Horizontal Difference'],
                                   mode='markers', 
                                   name=f'Cluster {cluster_id}'))

    fig.update_layout(title='3D K-Means Clustering of SLB Data',
                      scene=dict(
                          xaxis_title='Time (Seconds)',
                          yaxis_title='SLB Depth Difference',
                          zaxis_title='SLB Horizontal Difference'))

    st.plotly_chart(fig)

    cluster_formats = slb_data.groupby('Cluster')['Year/Mo. Category'].unique()
    for cluster_id, formats_in_cluster in cluster_formats.items():
        st.write(f"Formats in Cluster {cluster_id}:", formats_in_cluster)

def kmeans_by_total_variation(slb_data, features=['SLB Depth Difference', 'SLB Horizontal Difference'], k_values=range(1, 21)):
    slb_cluster_data = slb_data[features]

    scaler = StandardScaler()
    slb_cluster_data_scaled = scaler.fit_transform(slb_cluster_data)

    def calculate_total_variation(data, k_values):
        total_variation = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            closest, distances = pairwise_distances_argmin_min(data, kmeans.cluster_centers_)
            total_variation.append(sum(distances))
        return total_variation

    total_variation_values = calculate_total_variation(slb_cluster_data_scaled, k_values)

    st.write("### Elbow Curve for Total Variation")
    st.line_chart(pd.DataFrame({"k": k_values, "Total Variation": total_variation_values}).set_index("k"))

    optimal_k = np.argmin(np.diff(total_variation_values)) + 1
    st.write(f'Optimal number of clusters (k): {optimal_k}')

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    slb_data['Cluster'] = kmeans.fit_predict(slb_cluster_data_scaled)

    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    cluster_centers['Cluster'] = cluster_centers.index + 1
    st.write('\nCluster Centers:')
    st.write(cluster_centers)

    slb_data['Nearest Cluster Center'] = pairwise_distances_argmin_min(slb_cluster_data_scaled, kmeans.cluster_centers_)[0] + 1

    st.write('\nUpdated SLB Data with Clusters:')
    st.write(slb_data[['SLB origin time', 'SLB Depth Difference', 'Cluster', 'Nearest Cluster Center']].head())

def kmeans_by_wcss(slb_data_for_clustering, features=['SLB Depth Difference', 'SLB Horizontal Difference'], k_values=range(1, 11)):
    slb_cluster_data = slb_data_for_clustering[features]

    scaler = StandardScaler()
    slb_cluster_data_scaled = scaler.fit_transform(slb_cluster_data)

    def calculate_wcss(data, k_values):
        wcss = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
        return wcss

    wcss_values = calculate_wcss(slb_cluster_data_scaled, k_values)

    st.write("### Elbow Curve for WCSS")
    st.line_chart(pd.DataFrame({"k": k_values, "WCSS": wcss_values}).set_index("k"))

    optimal_k = np.argmin(np.diff(wcss_values)) + 1
    st.write(f'Optimal number of clusters (k): {optimal_k}')

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    slb_data_for_clustering['Cluster'] = kmeans.fit_predict(slb_cluster_data_scaled)

    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
    cluster_centers_df['Cluster'] = cluster_centers_df.index + 1
    st.write('\nCluster Centers:')
    st.write(cluster_centers_df)

    slb_data_for_clustering['Nearest Cluster Center'] = pairwise_distances_argmin_min(slb_cluster_data_scaled, kmeans.cluster_centers_)[0] + 1

    st.write('\nUpdated SLB Data with Clusters:')
    st.write(slb_data_for_clustering[['SLB Horizontal Difference', 'SLB Depth Difference', 'Cluster', 'Nearest Cluster Center']].head())

def custom_kmeans_clustering(slb_data, k_clusters, features=['SLB Depth Difference', 'SLB origin time']):
    scaler = StandardScaler()

    slb_data['SLB origin time'] = pd.to_datetime(slb_data['SLB origin time'])
    slb_data['Timestamp'] = slb_data['SLB origin time'].astype(np.int64) // 10**9

    timestamp_feature = slb_data['Timestamp']
    slb_data_without_timestamp = slb_data.drop(columns=['SLB origin time', 'Timestamp'])

    scaled_features = scaler.fit_transform(slb_data_without_timestamp[features[:-1]])

    scaled_features_with_timestamp = np.column_stack((scaled_features, timestamp_feature))

    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    slb_data['Cluster'] = kmeans.fit_predict(scaled_features_with_timestamp)

    return slb_data

def plot_normalized_time_clustering(slb_data, k_clusters):
    fig = go.Figure()

    for cluster_id in range(k_clusters):
        cluster_data = slb_data[slb_data['Cluster'] == cluster_id]
        fig.add_trace(go.Scatter(x=[cluster_id] * len(cluster_data), 
                                 y=cluster_data['SLB origin time'], 
                                 mode='markers', name=f'Cluster {cluster_id}',
                                 text=cluster_data['Year/Mo. Category'],  
                                 hoverinfo='text'))

    fig.update_layout(xaxis_title='Cluster', yaxis_title='SLB Origin Time', title='Normalized Time Clustering Results')
    st.plotly_chart(fig)
    
    cluster_formats = slb_data.groupby('Cluster')['Year/Mo. Category'].unique()
    for cluster_id, formats_in_cluster in cluster_formats.items():
        st.write(f"Formats in Cluster {cluster_id}:", formats_in_cluster)

def normalized_kmeansby_totalVariation(slb_data, k_clusters):

    slb_data['Normalized Time'] = (slb_data['SLB origin time'] - slb_data['SLB origin time'].min()) / (slb_data['SLB origin time'].max() - slb_data['SLB origin time'].min())
    features = ['Normalized Time', 'SLB Depth Difference']

    scaler = StandardScaler()
    slb_cluster_data_scaled = scaler.fit_transform(slb_data[features])

    def calculate_total_variation(data, k_values):
        total_variation = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            closest, distances = pairwise_distances_argmin_min(data, kmeans.cluster_centers_)
            total_variation.append(sum(distances))
        return total_variation

    k_values = range(1, 21)
    total_variation_values = calculate_total_variation(slb_cluster_data_scaled, k_values)

    st.write("### Elbow Curve for Total Variation")
    st.line_chart(pd.DataFrame({"k": k_values, "Total Variation": total_variation_values}).set_index("k"))

    optimal_k = np.argmin(np.diff(total_variation_values)) + 1
    st.write(f'Optimal number of clusters (k): {optimal_k}')

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    slb_data['Cluster'] = kmeans.fit_predict(slb_cluster_data_scaled)

    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    cluster_centers['Cluster'] = cluster_centers.index + 1

    st.write('\nCluster Centers:')
    st.write(cluster_centers)

    slb_data['Nearest Cluster Center'] = pairwise_distances_argmin_min(slb_cluster_data_scaled, kmeans.cluster_centers_)[0] + 1

    st.write('\nUpdated SLB Data with Clusters:')
    st.write(slb_data[['SLB origin time', 'SLB Depth Difference', 'Cluster', 'Nearest Cluster Center']].head())

def plot_depth_normalized_time_clustering(slb_data, k_clusters):
    fig = go.Figure()
    features = ['SLB Depth Difference', 'Normalized Time']
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    slb_data['Cluster'] = kmeans.fit_predict(slb_data[features])

    for cluster_id in range(k_clusters):
        cluster_data = slb_data[slb_data['Cluster'] == cluster_id]
        fig.add_trace(go.Scatter(x=cluster_data['SLB Depth Difference'], y=cluster_data['Normalized Time'], 
                                 mode='markers', name=f'Cluster {cluster_id}',
                                 text=cluster_data['Year/Mo. Category'],  
                                 hoverinfo='text'))

    fig.update_layout(title='K-Means Clustering of SLB Data',
                      xaxis_title='SLB Depth Difference', yaxis_title='Normalized Time')
    st.plotly_chart(fig)
    
    cluster_formats = slb_data.groupby('Cluster')['Year/Mo. Category'].unique()
    for cluster_id, formats_in_cluster in cluster_formats.items():
        st.write(f"Formats in Cluster {cluster_id}:", formats_in_cluster)

def plot_time_depth_clustering(slb_data, k_clusters):
    slb_data['SLB origin time'] = pd.to_datetime(slb_data['SLB origin time'])
    slb_data['TimeInSeconds'] = (slb_data['SLB origin time'] - slb_data['SLB origin time'].min()).dt.total_seconds() + 1
    features = ['TimeInSeconds', 'SLB Horizontal Difference', 'SLB Depth Difference']
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    slb_data['Cluster'] = kmeans.fit_predict(slb_data[features])

    fig = go.Figure()

    for cluster_id in range(k_clusters):
        cluster_data = slb_data[slb_data['Cluster'] == cluster_id]
        fig.add_trace(go.Scatter(x=cluster_data['TimeInSeconds'], y=cluster_data['SLB Depth Difference'],
                                mode='markers', name=f'Cluster {cluster_id}',
                                text=cluster_data['Year/Mo. Category'],  
                                hoverinfo='text')) 
    
    mountain_heights = {'Precambrian': -200, 'Argenta': 0, 'Mt Simon A Lower': 100, 'Mt Simon A Upper': 600, 'Mt Simon B': 800, 'Mt Simon C': 1000}
    for mountain, height in mountain_heights.items():
        fig.add_shape(type="line", x0=min(slb_data['TimeInSeconds']), y0=height, x1=max(slb_data['TimeInSeconds']), y1=height,
                      line=dict(color="red", width=2))
        fig.add_annotation(text=mountain, xref="x", yref="y", x=max(slb_data['TimeInSeconds']), y=height, showarrow=False, yshift=5, xshift=10)

    fig.update_layout(title='K-Means Clustering of SLB Data based on Time and Depth Difference',
                    xaxis_title='Time (Seconds)', yaxis_title='SLB Depth Difference',
                    legend_title='Cluster', showlegend=True,
                    height=600, width=900)

    st.plotly_chart(fig)

    # Add silhouette score and Davies-Bouldin index
    # Sample for silhouette score calculation to avoid MemoryError
    sample_size = min(1000, len(slb_data[features]))  # Use a smaller sample if the dataset is very large
    if len(slb_data[features]) > sample_size:
        sample_data = slb_data[features].sample(n=sample_size, random_state=42)
        sample_labels = slb_data.loc[sample_data.index, 'Cluster']
    else:
        sample_data = slb_data[features]
        sample_labels = slb_data['Cluster']

    silhouette_avg = silhouette_score(sample_data, sample_labels)
    db_index = davies_bouldin_score(slb_data[features], slb_data['Cluster'])
    st.write(f'Silhouette Score for Time vs Depth Clustering (sampled): {silhouette_avg}')
    st.write(f'Davies-Bouldin Index for Time vs Depth Clustering: {db_index}')
    
    cluster_formats = slb_data.groupby('Cluster')['Year/Mo. Category'].unique()
    for cluster_id, formats_in_cluster in cluster_formats.items():
        st.write(f"Formats in Cluster {cluster_id}:", formats_in_cluster)

def perform_clustering(slb_data, k_clusters):
    slb_data['SLB origin time'] = pd.to_datetime(slb_data['SLB origin time'])

    slb_data['Normalized Time'] = (slb_data['SLB origin time'] - slb_data['SLB origin time'].min()) / (
                slb_data['SLB origin time'].max() - slb_data['SLB origin time'].min())

    features = ['Normalized Time', 'SLB Horizontal Difference', 'SLB Depth Difference', 'SLB Total Difference']

    scaler = StandardScaler()
    slb_data_scaled = scaler.fit_transform(slb_data[features])

    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    slb_data['Cluster'] = kmeans.fit_predict(slb_data_scaled)

    return slb_data[['SLB origin time', 'Cluster']]

def plot_kmeans_horizontal_vs_time(slb_data, k_clusters):
    fig = go.Figure()
    features = ['SLB Horizontal Difference', 'TimeInSeconds']
    slb_data['SLB origin time'] = pd.to_datetime(slb_data['SLB origin time'])
    slb_data['TimeInSeconds'] = (slb_data['SLB origin time'] - slb_data['SLB origin time'].min()).dt.total_seconds()
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    slb_data['Cluster'] = kmeans.fit_predict(slb_data[features])

    for cluster_id in range(k_clusters):
        cluster_data = slb_data[slb_data['Cluster'] == cluster_id]
        fig.add_trace(go.Scatter(x=cluster_data['SLB Horizontal Difference'], y=cluster_data['TimeInSeconds'], 
                                 mode='markers', name=f'Cluster {cluster_id}',
                                 text=cluster_data['Year/Mo. Category'],  
                                 hoverinfo='text'))

    fig.update_layout(title='K-Means Clustering of SLB Data: Horizontal Difference vs Time',
                      xaxis_title='SLB Horizontal Difference', yaxis_title='Time (Seconds)')
    st.plotly_chart(fig)

    # Add silhouette score and Davies-Bouldin index
    # Sample for silhouette score calculation to avoid MemoryError
    sample_size = min(1000, len(slb_data[features]))  # Use a smaller sample if the dataset is very large
    if len(slb_data[features]) > sample_size:
        sample_data = slb_data[features].sample(n=sample_size, random_state=42)
        sample_labels = slb_data.loc[sample_data.index, 'Cluster']
    else:
        sample_data = slb_data[features]
        sample_labels = slb_data['Cluster']

    silhouette_avg = silhouette_score(sample_data, sample_labels)
    db_index = davies_bouldin_score(slb_data[features], slb_data['Cluster'])
    st.write(f'Silhouette Score for Horizontal Difference vs Time Clustering (sampled): {silhouette_avg}')
    st.write(f'Davies-Bouldin Index for Horizontal Difference vs Time Clustering: {db_index}')
    
    cluster_formats = slb_data.groupby('Cluster')['Year/Mo. Category'].unique()
    for cluster_id, formats_in_cluster in cluster_formats.items():
        st.write(f"Formats in Cluster {cluster_id}:", formats_in_cluster)

def plot_kmeans_normalized_horizontal_vs_time(slb_data, k_clusters):
    fig = go.Figure()
    slb_data['Timestamp'] = pd.to_datetime(slb_data['SLB origin time']).astype(np.int64) // 10**9
    slb_data['Normalized Time'] = (slb_data['Timestamp'] - slb_data['Timestamp'].min()) / (slb_data['Timestamp'].max() - slb_data['Timestamp'].min())
    slb_data['Normalized Horizontal Difference'] = (slb_data['SLB Horizontal Difference'] - slb_data['SLB Horizontal Difference'].min()) / (slb_data['SLB Horizontal Difference'].max() - slb_data['SLB Horizontal Difference'].min())
    features = ['Normalized Horizontal Difference', 'Normalized Time']
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    slb_data['Cluster'] = kmeans.fit_predict(slb_data[features])

    for cluster_id in range(k_clusters):
        cluster_data = slb_data[slb_data['Cluster'] == cluster_id]
        fig.add_trace(go.Scatter(x=cluster_data['Normalized Horizontal Difference'], y=cluster_data['Normalized Time'], 
                                 mode='markers', name=f'Cluster {cluster_id}',
                                 text=cluster_data['Year/Mo. Category'],  
                                 hoverinfo='text'))

    fig.update_layout(title='K-Means Clustering of SLB Data: Normalized Horizontal Difference vs Normalized Time',
                      xaxis_title='Normalized Horizontal Difference', yaxis_title='Normalized Time')
    st.plotly_chart(fig)
    
    cluster_formats = slb_data.groupby('Cluster')['Year/Mo. Category'].unique()
    for cluster_id, formats_in_cluster in cluster_formats.items():
        st.write(f"Formats in Cluster {cluster_id}:", formats_in_cluster)

def perform_dbscan_clustering(slb_data, eps=0.5, min_samples=5, features=['SLB Depth Difference', 'SLB Horizontal Difference']):
    st.markdown("""
    Performs DBSCAN clustering on the provided dataset.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm 
    that identifies clusters based on the density of data points. Unlike K-Means, DBSCAN does not require the 
    number of clusters to be specified beforehand and can find arbitrarily shaped clusters.

    **Parameters:**
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    """)
    st.write("Performing DBSCAN Clustering on Dataset...")

    slb_data['Year/Mo. Category'] = slb_data['Year/Mo. Category'].astype(str)

    data_for_clustering = slb_data[features].dropna()

    scaler = StandardScaler()
    data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    slb_data['cluster'] = dbscan.fit_predict(data_for_clustering_scaled)

    silhouette_avg = silhouette_score(data_for_clustering_scaled, slb_data['cluster'])
    st.write(f'Silhouette Score for DBSCAN: {silhouette_avg}')

    unique_clusters = slb_data['cluster'].unique()
    custom_colors = px.colors.qualitative.Plotly[:len(unique_clusters)]

    fig = go.Figure()

    for cluster_id in unique_clusters:
        cluster_data = slb_data[slb_data['cluster'] == cluster_id]
        fig.add_trace(go.Scatter(x=cluster_data['SLB Horizontal Difference'],
                                 y=cluster_data['SLB Depth Difference'],
                                 mode='markers',
                                 marker=dict(color=custom_colors[cluster_id % len(custom_colors)]),
                                 name=f'Cluster {cluster_id}'))

    fig.update_layout(title='DBSCAN Clustering of SLB Data',
                      xaxis_title='SLB Horizontal Difference',
                      yaxis_title='SLB Depth Difference')

    st.plotly_chart(fig)
    
    cluster_formats = slb_data.groupby('cluster')['Year/Mo. Category'].unique()
    for cluster_id, formats_in_cluster in cluster_formats.items():
        st.write(f"Formats in Cluster {cluster_id}:", formats_in_cluster)

def plot_k_distance_graph(data, k):
    st.markdown("""
    ### K-Distance Graph
    The K-Distance graph helps to determine the optimal value for `eps` in DBSCAN. 
    By plotting the distances of each point to its k-th nearest neighbor (sorted in ascending order), 
    you can identify a "knee" point in the graph. The "knee" point, where the graph shows a sharp change, 
    is a good candidate for the `eps` value.
    """)
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    distances = np.sort(distances[:, k-1], axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-Distance Graph')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    st.pyplot(plt)

tab_viz, tab_kmeans, tab_kmeans_3d, tab_dbscan, tab_tv, tab_normalize, tab_optimal_dbscan = st.tabs(["Data Visualization", "K-Means", "3D K-Means", "DBSCAN", "K-Means Optimal", "Normalized", "Optimal DBSCAN"])

with tab_viz:
    data_visualization_page()

with tab_kmeans:
    start_date, end_date = get_date_input("tab_kmeans")
    slb_data = load_slb_data_with_date_range(start_date, end_date)
    kmeans_clustering_page(start_date, end_date, "tab_kmeans", k_clusters)

with tab_kmeans_3d:
    start_date, end_date = get_date_input("tab_kmeans_3d")
    kmeans_3d_clustering_page(start_date, end_date, k_clusters)

with tab_dbscan:
    st.title('DBSCAN Clustering')
    eps = st.slider("Select epsilon (eps) value", min_value=0.1, max_value=2.0, step=0.1, value=0.5)
    min_samples = st.slider("Select min_samples value", min_value=1, max_value=20, step=1, value=5)
    start_date, end_date = get_date_input("tab_dbscan")
    slb_data_for_clustering = load_slb_data_with_date_range(start_date, end_date)
    perform_dbscan_clustering(slb_data_for_clustering, eps=eps, min_samples=min_samples)

with tab_tv:
    start_date, end_date = get_date_input("tab_tv")
    slb_data_for_clustering = load_slb_data_with_date_range(start_date, end_date)
    st.title("K-means Optimal")
    st.markdown("""
        Determining the optimal number of clusters (k) is crucial for effective K-Means clustering. \
        This code offers two common methods: Total Variation and WCSS (Within-Cluster Sum of Squares). 
        
        Both methods involve calculating a metric that reflects the compactness of clusters for different k values. The optimal k is then chosen at the point where adding more clusters leads to diminishing returns in terms of reducing the metric, visualized as an "elbow curve". Total Variation focuses on the overall spread within clusters, while WCSS measures the sum of squared distances between data points and their cluster centers. Choosing the appropriate method depends on your specific data and the emphasis you want to place on within-cluster variance or distance-based cluster compactness.
        """)
    clustering_method = st.selectbox('Choose K-means Optimal Clustering Method', ['Total Variation', 'WCSS'])
    if clustering_method == "Total Variation":
        kmeans_by_total_variation(slb_data_for_clustering, features=['SLB Depth Difference', 'SLB Horizontal Difference'])
    if clustering_method == "WCSS":
        kmeans_by_wcss(slb_data_for_clustering, features=['SLB Depth Difference', 'SLB Horizontal Difference'])

with tab_normalize:
    st.title('Data Clustering Analysis')
    st.markdown(""" 
        This code section enables interactive exploration of K-Means clustering on seismic line data. Users can specify the desired number of clusters and choose from various analysis methods:

        Total Variation and WCSS: These methods assess cluster compactness through different metrics, visualized as elbow curves, to determine the optimal number of clusters.
        Time and Depth Clustering: Groups data based on original time and depth features.
        Depth and Normalized Time/Normalized Time Clustering: Utilize combinations of depth and normalized time for clustering, offering different perspectives on the data.
        This interactive interface empowers users to experiment with diverse clustering approaches, gaining insights into various data relationships and selecting the most appropriate method for their specific analysis objectives.
                """)
    
    k_cluster_norm = st.slider("Select Number of Clusters (Normalized):", min_value=2, max_value=20, value=15)
    slb_data_for_clustering = load_slb_data_with_date_range(start_date, end_date)
    slb_data = custom_kmeans_clustering(slb_data_for_clustering, k_clusters)
    clustering_tab = st.selectbox("Choose Analysis", [ "Total Variation",\
                                                        "WCSS",\
                                                        "Time and Depth Clustering",\
                                                        "Depth and Normalized Time Clustering",\
                                                        "Normalized Time Clustering",\
                                                        "Horizontal Difference vs Time",\
                                                        "Normalized Horizontal Difference vs Normalized Time"
                                                        ])
    
    if clustering_tab == "Total Variation": 
        st.subheader("Total Variation")
        normalized_kmeansby_totalVariation(slb_data, k_cluster_norm)
    
    if clustering_tab == "WCSS": 
        st.subheader("WCSS")
        kmeans_by_wcss(slb_data_for_clustering, features=['SLB Depth Difference', 'SLB Horizontal Difference'])

    if clustering_tab == "Normalized Time Clustering": 
        st.subheader("Normalized Time Clustering")
        plot_normalized_time_clustering(slb_data, k_cluster_norm)

    if clustering_tab == "Time and Depth Clustering": 
        st.subheader("Time and Depth Clustering")
        plot_time_depth_clustering(slb_data, k_cluster_norm)

    if clustering_tab == "Depth and Normalized Time Clustering": 
        st.subheader("Depth and Normalized Time Clustering")
        perform_clustering(slb_data, k_cluster_norm)
        plot_depth_normalized_time_clustering(slb_data, k_cluster_norm)
    
    if clustering_tab == "Horizontal Difference vs Time":
        st.subheader("Horizontal Difference vs Time")
        plot_kmeans_horizontal_vs_time(slb_data, k_cluster_norm)
    
    if clustering_tab == "Normalized Horizontal Difference vs Normalized Time":
        st.subheader("Normalized Horizontal Difference vs Normalized Time")
        plot_kmeans_normalized_horizontal_vs_time(slb_data, k_cluster_norm)

with tab_optimal_dbscan:
    st.title('Optimal DBSCAN Parameters')
    
    # Load the data
    start_date, end_date = get_date_input("tab_optimal_dbscan")
    slb_data_for_clustering = load_slb_data_with_date_range(start_date, end_date)
    
    features = ['SLB Depth Difference', 'SLB Horizontal Difference']
    data_for_clustering = slb_data_for_clustering[features].dropna()
    
    # Standardize features
    scaler = StandardScaler()
    data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)
    
    st.subheader('K-Distance Graph')
    k_value = st.slider("Select k value for K-Distance Graph", min_value=1, max_value=20, value=5)
    plot_k_distance_graph(data_for_clustering_scaled, k_value)

    st.subheader('DBSCAN Silhouette Scores')
    eps_values = st.text_input("Enter eps values to test (comma-separated)", "0.1, 0.2, 0.3, 0.4, 0.5")
    min_samples_values = st.text_input("Enter min_samples values to test (comma-separated)", "3, 4, 5, 6, 7")

    eps_values = [float(eps) for eps in eps_values.split(',')]
    min_samples_values = [int(ms) for ms in min_samples_values.split(',')]
    
    results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(data_for_clustering_scaled)
            if len(set(clusters)) > 1:
                silhouette_avg = silhouette_score(data_for_clustering_scaled, clusters)
                results.append((eps, min_samples, silhouette_avg))
    
    results_df = pd.DataFrame(results, columns=['eps', 'min_samples', 'silhouette_score'])
    st.write("### Silhouette Scores for different parameter combinations")
    st.dataframe(results_df)
    
    optimal_params = results_df.loc[results_df['silhouette_score'].idxmax()]
    st.write(f"Optimal parameters: eps = {optimal_params['eps']}, min_samples = {optimal_params['min_samples']}, silhouette_score = {optimal_params['silhouette_score']}")
