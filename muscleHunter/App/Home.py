import base64
import streamlit as st
import pandas as pd
import sklearn
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


@st.cache_resource
def classify(labels, uploaded_file):
    results_percentage = {}
    results_classifications = {}

    if uploaded_file is not None:
        results_percentage = {"Imagem": [], "Musculo": [], "Tecido": [], "Outros": []}
        results_classifications = {"Imagem": [], "Classificacao": []}

        for i, file in enumerate(uploaded_file):
            # Resize the image
            img = Image.open(file)
            img = np.array(img)
            img = img.reshape(-1, 3)

            # Load the model kmeans_model.sav
            kmeans_model = pickle.load(open('kmeans_model.sav', 'rb'))

            # Classify the image
            classifications = kmeans_model.predict(img)
            
            results_classifications["Imagem"].append(file.name)
            results_classifications["Classificacao"].append(classifications)

            labeled_classifications = [labels[i] for i in classifications]
            
            # Append the classifications to the results
            results_percentage["Imagem"].append(file.name)
            for label in labels:
                results_percentage[label].append(labeled_classifications.count(label) / len(labeled_classifications))


        # Show the total number of images analyzed
        st.write("Imagens analisadas: ", len(uploaded_file))

        # Plot summary of the classifications
        df = pd.DataFrame(results_percentage)
        df.set_index("Imagem", inplace=True)
        
        # Stacked bar chart of each classification (muscle, tissue, other)
        st.bar_chart(df, color=["#FD94FD", "#7934E0", "#FEE0FE"])

    return results_classifications, results_percentage


st.set_page_config(page_title='Muscle Hunter', page_icon=':muscle:', layout='centered', initial_sidebar_state='auto')
st.sidebar.success('Bem-vindo ao Muscle Hunter! :muscle:')

st.write("""
# Muscle Hunter
Carregue uma ou mais imagens para começar a análise
""")

# Input images
uploaded_file = st.file_uploader("", type="jpg", accept_multiple_files=True)

labels = ["Musculo", "Tecido", "Outros"]
results_classifications, results_percentage = classify(labels, uploaded_file)

if uploaded_file:
    # Filter the images
    st.write("""
    ## Análises Individuais
    """)
    image_name = st.selectbox("Imagens", results_classifications["Imagem"])

    st.write("### Classificação de", image_name)
    # Search for the image in the uploaded files
    col1, col2 = st.columns(2)

    with col1:
        img = None
        for file in uploaded_file:
            if file.name == image_name:
                img = Image.open(file)
                st.image(img, caption="Imagem selecionada", use_column_width=True)
                break

    # Pie chart of the classifications
    classifications = results_classifications["Classificacao"][results_classifications["Imagem"].index(image_name)]
    classifications = [labels[i] for i in classifications]

    with col2:
        fig, ax = plt.subplots()
        ax.pie([classifications.count(label) for label in labels], labels=labels, autopct='%1.1f%%', colors=["#FD94FD", "#FEE0FE", "#7934E0"])
        ax.axis('equal')
        # transparent background
        fig.patch.set_alpha(0)
        st.pyplot(fig)

    
    # Download CSV of results_percentage
    st.write("""
    ## Exportar resultados (CSV)
    """)

    if st.button("Exportar resultados"):
        df = pd.DataFrame(results_percentage)
        st.write(df)
    
