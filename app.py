import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ----------- Configuration générale -----------
st.set_page_config(page_title="Analyse E-commerce", layout="wide", page_icon="🛒")
st.markdown(
    """
    <h1 style='text-align: center; color: #333;'>
        🛍️ Application d’Analyse de Données E-commerce
    </h1>
    """,
    unsafe_allow_html=True
)

# ----------- Affichage du logo -----------
image = Image.open("customer-data-management.jpg")
st.image(image, width=500, caption="🔍 Exploration des comportements clients")

# ----------- Chargement du fichier -----------
st.markdown("### 📁 Chargement du fichier")
uploaded_file = st.file_uploader("Chargez votre fichier Excel de ventes :", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name='Online Retail')
    st.success("✅ Jeu de données chargé avec succès !")

    # ----------- Nettoyage des données -----------
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # ----------- Création des onglets -----------
    tab1, tab2 = st.tabs(["📊 Modèles d’analyse", "📈 Statistiques descriptives"])

    # ----------- Analyse par modèle -----------
    with tab1:
        st.markdown("### 🧠 Choisissez un modèle d’analyse")
        choix = st.selectbox("Quel type d'analyse souhaitez-vous effectuer ?", [
            "🛍️ Règles d'association (FP-Growth)",
            "👥 Segmentation K-means",
            "📦 Segmentation RFM"
        ])

        if choix == "🛍️ Règles d'association (FP-Growth)":
            st.markdown("#### 🔗 Règles d'association avec FP-Growth")
            basket = (df[df['Country'] == 'United Kingdom']
                        .groupby(['InvoiceNo', 'Description'])['Quantity']
                        .sum().unstack().fillna(0))
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
            frequent_itemsets = fpgrowth(basket_sets, min_support=0.02, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

        elif choix == "👥 Segmentation K-means":
            st.markdown("#### 🎯 Segmentation client avec K-means")
            max_date = df['InvoiceDate'].max()
            rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (max_date - x.max()).days,
                'InvoiceNo': 'nunique',
                'TotalPrice': 'sum'
            }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            kmeans = KMeans(n_clusters=4, random_state=42)
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

            st.write("Aperçu des clusters :")
            st.dataframe(rfm.head(10))

            st.markdown("##### 📊 Répartition des clients par cluster")
            st.bar_chart(rfm['Cluster'].value_counts())

        elif choix == "📦 Segmentation RFM":
            st.markdown("#### 📦 Segmentation basée sur RFM (Recency, Frequency, Monetary)")
            max_date = df['InvoiceDate'].max()
            rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (max_date - x.max()).days,
                'InvoiceNo': 'nunique',
                'TotalPrice': 'sum'
            }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

            rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
            rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])
            rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

            st.dataframe(rfm.head(10))
            st.markdown("##### 🔢 Top 10 segments les plus fréquents")
            st.bar_chart(rfm['RFM_Segment'].value_counts().head(10))

        st.success("✅ Analyse terminée avec succès !")

    # ----------- Statistiques descriptives -----------
    with tab2:
        st.markdown("### 🧮 Statistiques descriptives")
        st.dataframe(df.describe())

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📌 Histogramme des prix unitaires")
            fig1, ax1 = plt.subplots()
            sns.histplot(df['UnitPrice'], bins=50, ax=ax1, color="skyblue")
            ax1.set_title("Distribution des prix unitaires")
            st.pyplot(fig1)

        with col2:
            st.markdown("#### 📌 Boxplot des quantités")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df['Quantity'], ax=ax2, color="lightgreen")
            ax2.set_title("Distribution des quantités")
            st.pyplot(fig2)

        st.markdown("#### 🌍 Top 10 des pays par volume de transactions")
        st.bar_chart(df['Country'].value_counts().head(10))

else:
    st.info("📤 Veuillez charger un fichier Excel pour démarrer l’analyse.")
