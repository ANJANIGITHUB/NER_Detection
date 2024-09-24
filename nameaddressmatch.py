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

# Function to match name and address with the dataframe
def match_name_address(df: pd.DataFrame, user_name: str) -> pd.DataFrame:
    try:
        if 'name' not in df.columns:
            st.error("DataFrame must contain 'name' columns")
            return pd.DataFrame()

        # Start the timer before executing the multithreaded task
        start_time = time.time()

        # Multithreading for faster similarity calculation
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda name: match_single_name(name, user_name), df['name']))

        # Assign results back to the DataFrame
        df['name_similarity'] = [similarity for _, similarity in results]

        # Filter records with a similarity score > 75%
        filtered_df = df[df['name_similarity'] > 0.75]

        # Calculate the total execution time in minutes
        end_time = time.time()
        execution_time_seconds = (end_time - start_time) 

        #Total Party Data to Search
        #st.write(f"Total Data Searched is {len(df)/1000}K")

        # Display the execution time
        #st.write(f"Execution Time: {execution_time_seconds:.2f} Seconds")
        
        return filtered_df[['name', 'name_similarity']]

    except Exception as e:
        st.error(f"Error in matching: {e}")
        return pd.DataFrame()

# Streamlit app
def main():
    logo = Image.open("minerva_logo.jpg")  # Replace with your logo path
    #st.image(logo, caption="Company Logo", width=700)
    st.image(logo, width=800)

    st.subheader("Let's verify you are not part of sanction entities")

    df = pd.read_csv("Entity_data.csv")

    user_name = st.text_input("Enter Your Name:")

    if st.button("Match"):
        if user_name:
            result_df = match_name_address(df, user_name)
            if not result_df.empty:
                st.write("Matching results (score > 75%):")
                st.dataframe(result_df.sort_values(by=['name_similarity'], ascending=False)[:5])
                #result_df.sort_values(by=['name_similarity'], ascending=False)
            else:
                st.write("No records found with a similarity score greater than 75%.")
        else:
            st.error("Please provide Party Name/Org to match.")

if __name__ == "__main__":
    main()
