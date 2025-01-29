import streamlit as st
import pandas as pd
from PIL import Image
from pyjarowinkler import distance  # Ensure you have installed pyjarowinkler
import re
from concurrent.futures import ThreadPoolExecutor
import time

# Function to preprocess text by making it lowercase and removing special characters
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Function to calculate Jaro-Winkler similarity
def jaro_winkler_similarity(s1: str, s2: str) -> float:
    try:
        s1 = preprocess_text(s1)
        s2 = preprocess_text(s2)
        return distance.get_jaro_distance(s1, s2, winkler=True)
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return 0.0

# Function to match name using multithreading
def match_single_name(name: str, user_name: str) -> tuple:
    similarity = jaro_winkler_similarity(name, user_name)
    return name, similarity

# Function to match name with the dataframe
def match_name_address(df: pd.DataFrame, user_name: str) -> pd.DataFrame:
    try:
        if 'name' not in df.columns:
            st.error("DataFrame must contain 'name' column")
            return pd.DataFrame()

        # Multithreading for faster similarity calculation
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda name: match_single_name(name, user_name), df['name']))

        # Assign results back to the DataFrame
        df['name_similarity'] = [similarity for _, similarity in results]

        # Filter records with a similarity score > 60%
        filtered_df = df[df['name_similarity'] > 0.60]

        return filtered_df[['name', 'name_similarity']]

    except Exception as e:
        st.error(f"Error in matching: {e}")
        return pd.DataFrame()

# Streamlit app
def main():
    logo = Image.open("minerva_logo.jpg")  # Replace with your logo path
    st.image(logo, width=800)

    # Load entity data
    df = pd.read_csv("Entity_data.csv")

    # Use session state to manage the name after save
    if 'saved_name' not in st.session_state:
        st.session_state.saved_name = ""
    
    st.subheader("Please Register Yourself")
    # Input fields for name and APMID
    name_input = st.text_input("Enter Your Name:")
    apmid_input = st.text_input("Enter Your APMID:")

    # Save button to register the name and APMID
    if st.button("Save"):
        if name_input and apmid_input:
            # Save the name to session state
            st.session_state.saved_name = name_input
            st.success(f"Name '{name_input}' and APMID '{apmid_input}' saved!")

    st.subheader("Let's verify if you are not part of sanctioned entities")

    # Use the saved name to populate the "Enter Your Name" field
    user_name = st.text_input("Your Name for Matching:", value=st.session_state.saved_name)

    # Reset the saved name after the match
    if st.button("Match"):
        if user_name:
            result_df = match_name_address(df, user_name)
            if not result_df.empty:
                st.write("Ooo... You matched with one of the sanctioned entities. Further investigation required (score > 85%):")
                st.dataframe(result_df[['name', 'name_similarity']].sort_values(by=['name_similarity'], ascending=False).reset_index(drop=True))
            else:
                st.write("Congratulations! You are not part of any sanctioned list.")
            
            # Reset the "Enter Your Name" field for next input
            st.session_state.saved_name = ""
        else:
            st.error("Please provide a name to match.")

if __name__ == "__main__":
    main()
