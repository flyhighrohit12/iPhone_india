import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# Set page config
st.set_page_config(page_title="Apple iPhones Sold in India", layout="wide")

# Custom CSS to improve aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #0068c9;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    h1, h2, h3 {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Sidebar
st.sidebar.title("Navigation")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your Apple iPhone dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if not df.empty:
        st.sidebar.success("File uploaded successfully!")
else:
    st.error("Please upload a CSV file to begin the analysis.")
    st.stop()

if not df.empty:
    page = st.sidebar.radio("Go to", ["Introduction", "Data Exploration", "Visualizations", "Insights and Recommendations", "Conclusion"])

    # Main content
    st.title("Apple iPhones Sold in India Analysis")

    if page == "Introduction":
        st.header("Introduction")
        st.write("""
        Welcome to the Apple iPhones Sold in India Analysis Dashboard! This app provides an in-depth exploration of 
        iPhone sales data in the Indian market. The dataset includes valuable insights into various iPhone models, 
        their prices, discounts, and customer ratings.

        Use the sidebar to navigate through different sections of the analysis. Each section offers interactive 
        visualizations and insights to help you understand the trends and patterns in the iPhone market in India.

        Let's begin our journey through the world of Apple iPhones in India!
        """)

        st.subheader("Dataset Overview")
        st.dataframe(df.head())
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    elif page == "Data Exploration":
        st.header("Data Exploration")

        st.subheader("Data Types and Basic Statistics")
        st.write(df.dtypes)
        st.write(df.describe())

        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values)
        st.write("Percentage of missing values:")
        st.write((missing_values / len(df)) * 100)

        st.subheader("Correlation Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough numeric columns for correlation analysis.")

    elif page == "Visualizations":
        st.header("Visualizations")

        # Price Distribution
        st.subheader("Price Distribution of iPhones")
        fig = px.histogram(df, x="Sale Price", nbins=20, title="Distribution of iPhone Prices")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Discount Percentage vs. Rating
        st.subheader("Discount Percentage vs. Star Rating")
        fig = px.scatter(df, x="Discount Percentage", y="Star Rating", 
                        color="Product Name", size="Number Of Ratings",
                        hover_data=["Sale Price"],
                        title="Discount Percentage vs. Star Rating")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Top 10 Most Rated iPhones
        st.subheader("Top 10 Most Rated iPhones")
        top_10_rated = df.nlargest(10, "Number Of Ratings")
        fig = px.bar(top_10_rated, x="Product Name", y="Number Of Ratings", 
                    color="Star Rating", title="Top 10 Most Rated iPhones")
        fig.update_layout(xaxis_tickangle=-45, height=600)
        st.plotly_chart(fig, use_container_width=True)

        # RAM vs. Price
        st.subheader("RAM vs. Price")
        fig = px.scatter(df, x="Ram", y="Sale Price", color="Product Name", 
                        size="Number Of Ratings", hover_data=["Star Rating"],
                        title="RAM vs. Price")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Price Comparison by Storage Capacity
        st.subheader("Price Comparison by Storage Capacity")
        df['Storage'] = df['Product Name'].str.extract('(\d+\s*GB)')
        df['Storage'] = pd.to_numeric(df['Storage'].str.replace('GB', '').str.strip())
        fig = px.box(df, x="Storage", y="Sale Price", color="Storage",
                    title="Price Distribution by Storage Capacity")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Star Rating vs. Number of Ratings
        st.subheader("Star Rating vs. Number of Ratings")
        fig = px.scatter(df, x="Number Of Ratings", y="Star Rating", 
                        color="Product Name", size="Sale Price",
                        hover_data=["Discount Percentage"],
                        title="Star Rating vs. Number of Ratings")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Discount Percentage Distribution
        st.subheader("Discount Percentage Distribution")
        fig = px.histogram(df, x="Discount Percentage", nbins=20,
                        title="Distribution of Discount Percentages")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Price Trends Across iPhone Generations
        st.subheader("Price Trends Across iPhone Generations")
        df['Generation'] = df['Product Name'].str.extract('(\d+)').astype(float)
        avg_prices = df.groupby('Generation')['Sale Price'].mean().reset_index()
        fig = px.line(avg_prices, x="Generation", y="Sale Price", 
                    title="Average Price Trend Across iPhone Generations")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Feature Comparison: RAM, Storage, Price
        st.subheader("Feature Comparison: RAM, Storage, Price")
        fig = px.scatter_3d(df, x='Ram', y='Storage', z='Sale Price',
                            color='Product Name', size='Number Of Ratings',
                            hover_data=['Star Rating', 'Discount Percentage'],
                            title="3D Comparison: RAM, Storage, and Price")
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        # Word Cloud of Product Names
        st.subheader("Word Cloud of iPhone Models")
        text = " ".join(df['Product Name'])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    elif page == "Insights and Recommendations":
        st.header("Insights and Recommendations")

        st.write("""
        Based on the analysis of the Apple iPhones sold in India dataset, here are key insights and recommendations:

        1. **Price Range Diversity**:
           - iPhones in India are available across a wide price range, catering to different market segments.
           - Recommendation: Continue offering diverse price points to capture various consumer segments.

        2. **Discount Strategy**:
           - There's no strong correlation between discount percentage and star rating, suggesting that discounts alone don't drive customer satisfaction.
           - Recommendation: Focus on overall value proposition rather than relying heavily on discounts.

        3. **Popular Models**:
           - Certain iPhone models receive significantly more ratings, indicating higher popularity or sales volume.
           - Recommendation: Analyze the features of these popular models to inform future product development and marketing strategies.

        4. **RAM and Pricing**:
           - There's a general trend of higher prices for models with more RAM, but with significant variation.
           - Recommendation: Clearly communicate the value of higher RAM to justify price differences.

        5. **Customer Ratings**:
           - Most iPhones have high star ratings, indicating general customer satisfaction.
           - Recommendation: Maintain high quality standards and continue to focus on customer experience.

        6. **Storage Options**:
           - Different storage capacities significantly impact pricing.
           - Recommendation: Educate customers on the benefits of different storage options to help them make informed decisions.

        These insights suggest that Apple's iPhone lineup in India is diverse and generally well-received. To maintain and improve market position, Apple should focus on clear value communication, targeted marketing for popular models, and maintaining high quality standards across all price points.
        """)

    else:  # Conclusion
        st.header("Conclusion")
        st.write("""
        The analysis of Apple iPhones sold in India reveals a complex and dynamic market. 
        Key takeaways include:

        1. Wide range of products catering to various price segments
        2. High overall customer satisfaction as reflected in star ratings
        3. Varying popularity among different iPhone models
        4. Price influenced by factors like RAM and storage capacity

        Apple's strategy in India should focus on maintaining product diversity, clear communication of value, and 
        continuous improvement in customer experience to solidify its position in this important market.
        """)

    # Add a footer
    st.markdown("---")
    st.markdown("Created with ❤️ using Streamlit")

else:
    st.error("No data available. Please check your dataset or try uploading a different file.")