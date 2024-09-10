import streamlit as st
import pandas as pd
from PIL import Image
from pyjarowinkler import distance  # Ensure you have installed pyjarowinkler
import re

# Function to preprocess text by making it lowercase and removing special characters
def preprocess_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # Remove special characters using regex
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Function to calculate Jaro-Winkler similarity
def jaro_winkler_similarity(s1: str, s2: str) -> float:
    try:
        # Preprocess both strings before comparison
        s1 = preprocess_text(s1)
        s2 = preprocess_text(s2)
        return distance.get_jaro_distance(s1, s2, winkler=True)
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return 0.0

# Function to match name and address with the dataframe
def match_name_address(df: pd.DataFrame, user_name: str, user_address: str) -> pd.DataFrame:
    try:
        # Check if the necessary columns exist
        if 'name' not in df.columns or 'address' not in df.columns:
            st.error("DataFrame must contain 'name' and 'address' columns.")
            return pd.DataFrame()

        # Calculate similarity scores after preprocessing name and address
        df['name_similarity'] = df['name'].apply(lambda x: jaro_winkler_similarity(x, user_name))
        df['address_similarity'] = df['address'].apply(lambda x: jaro_winkler_similarity(x, user_address))
        
        # Combine the similarity scores for a final score
        df['combined_similarity'] = (df['name_similarity'] + df['address_similarity']) / 2
        
        # Filter records where combined similarity score is more than 75%
        filtered_df = df[df['combined_similarity'] > 0.75]
        
        return filtered_df[['name', 'address', 'name_similarity', 'address_similarity', 'combined_similarity']]
    
    except Exception as e:
        st.error(f"Error in matching: {e}")
        return pd.DataFrame()

# Streamlit app
def main():
    
    # Load the logo image
    logo = Image.open("maersk_pic2.jpg")  # Replace with your logo path

    # Display the logo at the top, resizing it to a default size (e.g., 300px width)
    st.image(logo, caption="Company Logo", width=700)

    st.title("Minerva Name and Address Matching")


    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with Sanctioned Lists", type=["csv"])

    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        #st.write("DataFrame loaded successfully:")
        #st.dataframe(df)
        
        # User input for name and address
        user_name = st.text_input("Party name:")
        user_address = st.text_input("Party address:")
        
        # Match button
        if st.button("Match"):
            if user_name and user_address:
                # Perform matching
                result_df = match_name_address(df, user_name, user_address)
                
                # Display results
                if not result_df.empty:
                    st.write("Matching results (score > 75%):")
                    st.dataframe(result_df)
                else:
                    st.write("No records found with a similarity score greater than 75%.")
            else:
                st.error("Please provide both name and address to match.")

if __name__ == "__main__":
    main()
