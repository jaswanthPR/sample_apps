import os
import streamlit as st
import pandas as pd
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import shutil

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF file {pdf_path}: {e}")
        return ""

# Function to extract text from Excel files
def extract_text_from_excel(excel_path, start_row=None, end_row=None):
    try:
        excel_data = pd.read_excel(excel_path)
        if start_row is None:
            start_row = 0
        if end_row is None:
            end_row = len(excel_data)
        text = " ".join(str(cell) for cell in excel_data.iloc[start_row:end_row].values.flatten())
        return text
    except Exception as e:
        st.error(f"Error reading Excel file {excel_path}: {e}")
        return ""

# Function to calculate cosine similarity between two texts
def calculate_similarity(file1_text, file2_text):
    try:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([file1_text, file2_text])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
        return cosine_sim[0][1]
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return 0.0

# Function to find similar files between selected files
def find_similar_files(file_texts, selected_files):
    similarity_matrix = pd.DataFrame(index=selected_files, columns=selected_files)

    for i in range(len(selected_files)):
        for j in range(i + 1, len(selected_files)):
            file1, file2 = selected_files[i], selected_files[j]
            similarity = calculate_similarity(file_texts[file1], file_texts[file2])
            similarity_matrix.loc[file1, file2] = similarity
            similarity_matrix.loc[file2, file1] = similarity

    return similarity_matrix

def main():
    st.title("File Similarity Checker")

    # Step 1: Upload files
    uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
    if uploaded_files:
        temp_dir = "temp_uploaded_files"
        os.makedirs(temp_dir, exist_ok=True)

        file_texts = {}
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            if file_path.lower().endswith('.pdf'):
                file_texts[uploaded_file.name] = extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
                file_texts[uploaded_file.name] = extract_text_from_excel(file_path)
            else:
                st.warning(f"Unsupported file format: {uploaded_file.name}")
                file_texts[uploaded_file.name] = ""

        st.success("Files uploaded successfully!")

        # Step 2: Specify the number of files to compare
        if len(file_texts) >= 2:
            num_files_to_compare = st.number_input("Enter the number of files to compare", min_value=2, max_value=len(file_texts), step=1)

            # Step 3: Select the files to compare
            if num_files_to_compare:
                selected_files = st.multiselect(f"Select {num_files_to_compare} files to compare", list(file_texts.keys()), max_selections=num_files_to_compare)

                if len(selected_files) == num_files_to_compare:
                    with st.spinner("Calculating similarities..."):
                        similarity_matrix = find_similar_files(file_texts, selected_files)

                    if not similarity_matrix.empty:
                        st.write("Similarity Matrix:")
                        st.dataframe(similarity_matrix)

        else:
            st.warning("Please upload at least two files for comparison.")

        # Option to delete previous memory
        if st.button("Delete Previous Memory"):
            shutil.rmtree(temp_dir)
            st.success("Previous memory deleted.")

if __name__ == "__main__":
    main()
