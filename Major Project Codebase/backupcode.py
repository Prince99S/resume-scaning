import streamlit as st
import pickle
import re
import nltk
import base64

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to your local background image
image_path = r"./background.jpg"  # Update this to your local image path
base64_image = get_base64_image(image_path)

# Web app
def main():
    # Add custom CSS with base64-encoded background image
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{base64_image}");
                background-size: cover;
                background-position: center;
            }}
            .title {{
                text-align: center;
                font-size: 40px;
                color: #000000;
                margin-bottom: 20px;
            }}
            .result {{
                # text-align: center;
                font-size: 20px;
                background-color: #ffffff;
                margin-top: 10px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="title">Career Counselling</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader('Upload Resume', type=['doc', 'docx', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        # Add a submit button
        if st.button("Submit"):
            cleaned_resume = clean_resume(resume_text)
            input_features = tfidfd.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")
            # Display both predicted ID and category in the same section
            result_message = f"Predicted ID: {prediction_id} </br> Predicted Category: {category_name}"
            st.markdown(f'<div class="result">{result_message}</div>', unsafe_allow_html=True)


# Python main
if __name__ == "__main__":
    main()
