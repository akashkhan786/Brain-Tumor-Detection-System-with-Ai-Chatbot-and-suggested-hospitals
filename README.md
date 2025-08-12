
 **Brain Tumor Detection System with AI Chatbot & Hospital Suggestions**

**Note:** According to the file uploading guidelines, large files such as `prediction_model.keras` (\~162 MB) and the `vectorstore` memory file for the chatbot (\~200 MB) have been **deleted** from this version. The full code for both the AI prediction model and the chatbot vector memory logic is still present in the repository.

**Project Overview**

This project is an AI-powered medical assistant designed for brain tumor detection and post-diagnosis guidance. It includes:

Tumor Classification Model using VGG16
AI Chatbot using a Mistral 7B hybrid LLM model
Suggested Hospital System based on fuzzy search from CSV data

**Tumor Detection**

**Model:** Pre-trained VGG16 CNN
**Dataset**: [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
**Classes:** Glioma, Meningioma, Pituitary, No Tumor
**Image Count:** 7,022 MRI images
**Test Accuracy:** 98.12%


 **AI Chatbot**

**Model Used:** Mistral 7B (Hybrid LLM)

**Brain Tumor Knowledge Base:**
   Data extracted using RAG (Retrieval-Augmented Generation)
   \~350+ PDFs sourced from PubMed and Google (2024–2025)
   Hospital Data Source:**

   **CSV files**
   Free public data using fuzzy search and query handling
  **Hospital**

 In chabot you can easily asked about hospitals and you can filter out in query some specific data like (_ hospitals contact no ? ,where is  _hospitals ? , is in this city any hospitals ?) and fuzzy query matching
 Hospital data is stored in `.csv` files directly loaded into the system



 **Requirements**

All dependencies are listed in `requirements.txt`.

 Before You Run:

1. Download and add the following **manually**:

    `model link below` (VGG16 model weights)
    `vectorstore` memory/database for the chatbot link below
2. Ensure all required Python packages are installed:

   
   pip install -r requirements.txt
   


 **Folder Contents**
In main folder fyp_project there is :
app : coding
static : images
uploads : To upload images for prediction
templates : frontend template
model : predicted model 

 Chatbot/ – Code for VGG16-based tumor classification
 Chatbot/ – Code for AI chatbot using Mistral 7B
 Chatbot/data/ – CSV files for hospital information
 requirements.txt – Python packages needed
 README.md – This file

   **Dataset for clases prediction**
   link:https://drive.google.com/drive/folders/1CTMBVnEqq04-94y9M9zPmTMPMtpQhIBD?usp=drive_link

   
   **Model**
   link :https://drive.google.com/file/d/1caDks3dokBQgyc9eMu04gwGsSG47g5nE/view?usp=drive_link

   
   **Data for chatbot**
   link:https://drive.google.com/drive/folders/1f-GdwDetwcG_P3TibDVZAXefak5wS1hE?usp=drive_link
   
  **How to Run**

1. Ensure you have the full files mentioned above.
2. Run the tumor detection model or start the chatbot system as per instructions in the respective code files.



  **Contact**
For further queries ....
Whatsupp Only :+923119688358
syedshakeelbacha@gmail.com


