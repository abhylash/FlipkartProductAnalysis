import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go

# Set Page Config with more modern settings
st.set_page_config(
    page_title="E-Commerce Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🛍️"
)

# Enhanced CSS for a more modern look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f9fafb;
        padding: 1rem;
    }
    
    /* Header styling */
    .dashboard-header {
        padding: 1.5rem 0;
        text-align: center;
        margin-bottom: 2rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #4361EE, #3A0CA3);
        color: white;
    }
    
    /* Card styling */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric values */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3A0CA3;
        margin-bottom: 0.5rem;
    }
    
    /* Metric labels */
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        padding: 1rem 0;
        margin: 1.5rem 0 1rem 0;
        color: #1E3A8A;
        border-bottom: 2px solid #E5E7EB;
        font-weight: 600;
    }
    
    /* Chart containers */
    .chart-container {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        color: #6B7280;
        border-top: 1px solid #E5E7EB;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4361EE;
        color: white;
        border-radius: 8px;
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        background-color: white;
        color: black;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: black;
        font-weight: bold;
    }
    
    /* Filter section styling */
    .filter-container {
        background-color: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .filter-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #3A0CA3;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("flipkartupdated.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'flipkartupdated.csv' is in the correct location.")
        return pd.DataFrame()  # Return empty DataFrame to avoid errors

df = load_data()

# Check if DataFrame is empty
if df.empty:
    st.error("No data available. Please check your dataset file.")
    st.stop()

# Check for required columns
expected_columns = ["Product_name", "Price_cleaned", "Rate", "Review_cleaned"]
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    st.error(f"⚠️ Missing columns in dataset: {missing_columns}")
    st.stop()

# Data cleaning
df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
df["Price_cleaned"] = pd.to_numeric(df["Price_cleaned"], errors="coerce")
df.dropna(subset=["Rate", "Price_cleaned"], inplace=True)

# Dashboard header
st.markdown("""
<div class="dashboard-header">
    <h1>🛍️ Flipkart Product Analysis Dashboard</h1>
    <p>Comprehensive insights into product performance, pricing trends, and customer sentiment</p>
</div>
""", unsafe_allow_html=True)

# Filters in the main page
st.markdown('<div class="filter-container">', unsafe_allow_html=True)
st.markdown('<div class="filter-title">Dashboard Filters</div>', unsafe_allow_html=True)

# Create inline columns for filters
col1, col2 = st.columns(2)

with col1:
    price_range = st.slider(
        "Price Range (₹)",
        min_value=int(df["Price_cleaned"].min()),
        max_value=int(df["Price_cleaned"].max()),
        value=(int(df["Price_cleaned"].min()), int(df["Price_cleaned"].max()))
    )

with col2:
    rating_filter = st.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.5)

st.markdown('</div>', unsafe_allow_html=True)

# Apply filters
filtered_df = df[(df["Price_cleaned"] >= price_range[0]) & 
                (df["Price_cleaned"] <= price_range[1]) &
                (df["Rate"] >= rating_filter)]

# Create tabs for better organization
tabs = st.tabs(["Overview", "Price Analysis", "Sentiment Analysis", "Top Products"])

# OVERVIEW TAB
with tabs[0]:
    st.markdown('<div class="section-header"><h2>📊 Key Metrics</h2></div>', unsafe_allow_html=True)
    
    # Key metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Products</div>
                <div class="metric-value">{filtered_df["Product_name"].nunique()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Price</div>
                <div class="metric-value">₹{filtered_df['Price_cleaned'].mean():,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Rating</div>
                <div class="metric-value">{filtered_df['Rate'].mean():.2f} ★</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col4:
        # Calculate sentiment more efficiently
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            with st.spinner('Downloading NLTK resources...'):
                nltk.download('vader_lexicon', quiet=True)
        
        sia = SentimentIntensityAnalyzer()
        
        # Calculate sentiment for filtered data
        if 'Sentiment' not in filtered_df.columns:
            filtered_df["Sentiment"] = filtered_df["Review_cleaned"].astype(str).apply(
                lambda text: "Positive" if sia.polarity_scores(text)["compound"] > 0.05 
                else "Negative" if sia.polarity_scores(text)["compound"] < -0.05 
                else "Neutral"
            )
        
        positive_pct = (filtered_df["Sentiment"] == "Positive").mean() * 100
        
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Positive Reviews</div>
                <div class="metric-value">{positive_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Overview charts
    st.markdown('<div class="section-header"><h2>📈 Overview Charts</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = px.histogram(
            filtered_df, 
            x="Price_cleaned", 
            nbins=50, 
            title="Product Price Distribution",
            labels={"Price_cleaned": "Price (₹)", "count": "Number of Products"},
            color_discrete_sequence=["#4361EE"]
        )
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#E5E7EB"),
            yaxis=dict(gridcolor="#E5E7EB")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = px.scatter(
            filtered_df, 
            x="Price_cleaned", 
            y="Rate", 
            title="Price vs. Rating",
            labels={"Price_cleaned": "Price (₹)", "Rate": "Rating"},
            color_discrete_sequence=["#4361EE"],
            opacity=0.6
        )
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#E5E7EB"),
            yaxis=dict(gridcolor="#E5E7EB")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# PRICE ANALYSIS TAB
with tabs[1]:
    st.markdown('<div class="section-header"><h2>💰 Price Analysis</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Price distribution with Plotly for better interactivity
        fig = px.box(
            filtered_df, 
            y="Price_cleaned", 
            title="Price Distribution",
            labels={"Price_cleaned": "Price (₹)"},
            color_discrete_sequence=["#4361EE"]
        )
        fig.update_layout(
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#E5E7EB")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Create price bins for visualization
        price_bins = [0, 1000, 5000, 10000, 50000, float(filtered_df["Price_cleaned"].max()) + 1]
        price_labels = ['< ₹1K', '₹1K-₹5K', '₹5K-₹10K', '₹10K-₹50K', '> ₹50K']

        # Make sure bins are sorted and unique
        price_bins = sorted(list(set(price_bins)))
        
        # Fix price labels if needed
        if len(price_bins) - 1 != len(price_labels):
            price_labels = [f'Bin {i+1}' for i in range(len(price_bins)-1)]

        filtered_df['price_category'] = pd.cut(
            filtered_df['Price_cleaned'], 
            bins=price_bins, 
            labels=price_labels
        )
        
        price_category_counts = filtered_df['price_category'].value_counts().reset_index()
        price_category_counts.columns = ['price_category', 'count']
        
        fig = px.pie(
            price_category_counts, 
            values='count', 
            names='price_category',
            title='Price Range Distribution',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Price prediction model
    st.markdown('<div class="section-header"><h2>🔮 Price Prediction Model</h2></div>', unsafe_allow_html=True)
    
    # Train Random Forest Model for Price Prediction
    if len(filtered_df) > 10:  # Only run the model if we have enough data
        X = filtered_df[["Rate"]]
        y = filtered_df["Price_cleaned"]
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">RMSE</div>
                        <div class="metric-value">₹{rmse:.2f}</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Lower is better</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value">{r2:.2f}</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Higher is better (max 1.0)</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error building prediction model: {str(e)}")
    else:
        st.warning("Not enough data for price prediction model. Please adjust filters.")

# SENTIMENT ANALYSIS TAB
with tabs[2]:
    st.markdown('<div class="section-header"><h2>💬 Sentiment Analysis</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        
        # Ensure proper order
        sentiment_order = ['Positive', 'Neutral', 'Negative']
        sentiment_counts['sentiment'] = pd.Categorical(
            sentiment_counts['sentiment'], 
            categories=sentiment_order, 
            ordered=True
        )
        sentiment_counts = sentiment_counts.sort_values('sentiment')
        
        # Create custom colors
        colors = {"Positive": "#4CAF50", "Neutral": "#9E9E9E", "Negative": "#F44336"}
        
        fig = px.bar(
            sentiment_counts, 
            x='sentiment', 
            y='count',
            title="Review Sentiment Distribution",
            labels={"sentiment": "Sentiment", "count": "Number of Reviews"},
            color='sentiment',
            color_discrete_map=colors
        )
        
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#E5E7EB"),
            yaxis=dict(gridcolor="#E5E7EB")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Calculate average rating by sentiment
        avg_rating = filtered_df.groupby("Sentiment")["Rate"].mean().reset_index()
        
        # Ensure proper order
        avg_rating['Sentiment'] = pd.Categorical(
            avg_rating['Sentiment'], 
            categories=sentiment_order, 
            ordered=True
        )
        avg_rating = avg_rating.sort_values('Sentiment')
        
        fig = px.bar(
            avg_rating,
            x="Sentiment",
            y="Rate",
            title="Average Rating by Sentiment",
            labels={"Sentiment": "Sentiment", "Rate": "Average Rating"},
            color="Sentiment",
            color_discrete_map=colors
        )
        
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#E5E7EB"),
            yaxis=dict(gridcolor="#E5E7EB"),
            yaxis_range=[0, 5]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Word clouds with better styling
    st.markdown('<div class="section-header"><h2>☁️ Word Clouds</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Initialize the wordclouds if there are reviews
    positive_reviews = " ".join(filtered_df[filtered_df["Sentiment"] == "Positive"]["Review_cleaned"].astype(str))
    negative_reviews = " ".join(filtered_df[filtered_df["Sentiment"] == "Negative"]["Review_cleaned"].astype(str))
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Positive Reviews")
        if len(positive_reviews) > 10:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color="white", 
                    max_words=100,
                    colormap="viridis",
                    contour_width=1,
                    contour_color='#4361EE'
                ).generate(positive_reviews)
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating positive word cloud: {str(e)}")
        else:
            st.info("Not enough positive reviews for word cloud")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Negative Reviews")
        if len(negative_reviews) > 10:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color="white", 
                    colormap="Reds", 
                    max_words=100,
                    contour_width=1,
                    contour_color='#F44336'
                ).generate(negative_reviews)
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating negative word cloud: {str(e)}")
        else:
            st.info("Not enough negative reviews for word cloud")
        st.markdown('</div>', unsafe_allow_html=True)

# TOP PRODUCTS TAB
with tabs[3]:
    st.markdown('<div class="section-header"><h2>🏆 Top Rated Products</h2></div>', unsafe_allow_html=True)
    
    # Top rated products
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Get top products by rating with at least 2 reviews
    product_counts = filtered_df.groupby("Product_name").size().reset_index(name="count")
    products_with_multiple_reviews = product_counts[product_counts["count"] >= 2]["Product_name"].tolist()
    
    # Filter for products with multiple reviews
    multi_review_df = filtered_df[filtered_df["Product_name"].isin(products_with_multiple_reviews)]
    
    # Get top rated products
    top_rated_products = multi_review_df.groupby("Product_name")["Rate"].mean().reset_index()
    top_rated_products = top_rated_products[top_rated_products["Product_name"].str.len() < 50]  # Filter out overly long names
    top_rated_products = top_rated_products.sort_values(by="Rate", ascending=False).head(10)
    
    fig = px.bar(
        top_rated_products,
        y="Product_name",
        x="Rate",
        title="Top 10 Highest Rated Products",
        labels={"Rate": "Average Rating", "Product_name": "Product"},
        color="Rate",
        color_continuous_scale="viridis",
        orientation='h'
    )
    
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#E5E7EB", range=[0, 5]),
        yaxis=dict(categoryorder='total ascending')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # NEW: Lowest rated products
    st.markdown('<div class="section-header"><h2>⚠️ Lowest Rated Products</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Get lowest rated products with at least 2 reviews
    lowest_rated_products = multi_review_df.groupby("Product_name")["Rate"].mean().reset_index()
    lowest_rated_products = lowest_rated_products[lowest_rated_products["Product_name"].str.len() < 50]
    lowest_rated_products = lowest_rated_products.sort_values(by="Rate", ascending=True).head(10)
    
    fig = px.bar(
        lowest_rated_products,
        y="Product_name",
        x="Rate",
        title="Top 10 Lowest Rated Products",
        labels={"Rate": "Average Rating", "Product_name": "Product"},
        color="Rate",
        color_continuous_scale="RdYlGn",
        orientation='h'
    )
    
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#E5E7EB", range=[0, 5]),
        yaxis=dict(categoryorder='total ascending')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Price-Rating bubble chart
    st.markdown('<div class="section-header"><h2>📊 Product Comparison</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Product metrics aggregation - fix any potential issues
    try:
        product_metrics = filtered_df.groupby("Product_name").agg({
            "Price_cleaned": "mean",
            "Rate": "mean",
        }).reset_index()
        
        # Add count as a separate step to avoid naming conflicts
        product_counts = filtered_df.groupby("Product_name").size().reset_index(name="count")
        product_metrics = product_metrics.merge(product_counts, on="Product_name")
        
        # Filter to products with multiple reviews
        product_metrics = product_metrics[product_metrics["count"] > 1]
        
        # Limit to reasonable length names
        product_metrics = product_metrics[product_metrics["Product_name"].str.len() < 50]
        
        # Take top 50 by count for visualization
        product_metrics = product_metrics.sort_values("count", ascending=False).head(50)
        
        fig = px.scatter(
            product_metrics,
            x="Price_cleaned",
            y="Rate",
            size="count",
            color="count",
            hover_name="Product_name",
            title="Product Rating vs Price (bubble size = number of reviews)",
            labels={"Price_cleaned": "Price (₹)", "Rate": "Rating", "count": "Number of Reviews"},
            color_continuous_scale="viridis",
            size_max=50
        )
        
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#E5E7EB"),
            yaxis=dict(gridcolor="#E5E7EB", range=[0, 5])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating comparison chart: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>📌 <b>E-Commerce Analytics Dashboard</b> | Built with Streamlit | By Abhilash Satish Wadekar</p>
    <p style="font-size: 0.8rem;">Last updated: March 2025</p>
</div>
""", unsafe_allow_html=True)
