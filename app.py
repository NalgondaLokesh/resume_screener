import nltk
import re
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import PyPDF2

nltk.download('punkt')
nltk.download('stopwords')
clf = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))

def clean_text(text):
  text = re.sub('http\S+\s*', ' ', text)  # Remove URLs
  text = re.sub('RT|cc', ' ', text)  # Remove RT and cc
  text = re.sub('#\S+', '', text)  # Remove hashtags
  text = re.sub('@\S+', '  ', text)  # Remove mentions
  text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # Remove punctuations
  text = re.sub(r'[^\x00-\x7f]',r' ', text) # Remove non-ASCII characters
  text = re.sub('\s+', ' ', text)  # Remove extra whitespace
  return text

#Function to predict the category of the resume
def pred(input_resume):
  cleaned_text = clean_text(input_resume) # Changed cleanResume to clean_text
  vectorized_text = tfidf.transform([cleaned_text])
  vectorized_text = vectorized_text.toarray()
  predicted_category = clf.predict(vectorized_text)
  prediction = le.inverse_transform(predicted_category)
  return prediction[0]

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text



#web app
def main():
    st.title("Resume Screening App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")
    upload_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

    if upload_file is not None:
        try:
            resume_text = handle_file_upload(upload_file)
            st.write("Successfully extracted the text from the uploaded resume.")
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            st.write("Please ensure the file is a valid PDF, DOCX, or TXT file and try again.")

if __name__ == '__main__':
    main()
