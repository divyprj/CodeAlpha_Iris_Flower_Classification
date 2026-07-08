import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import os
from PIL import Image

# Import ML model logic
from model import load_data, train_model, predict_species

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling — targets modern Streamlit CSS selectors
st.markdown("""
<style>
    /* ===================== GLOBAL DARK THEME ===================== */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }

    /* ===================== SIDEBAR DARK THEME ===================== */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
    }

    /* ===================== HEADINGS ===================== */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }

    /* ===================== ALL WIDGET LABELS ===================== */
    /* Selectbox, slider, number_input, checkbox, radio labels */
    label, .stSlider label, .stSelectbox label, .stNumberInput label,
    .stCheckbox label, .stRadio label,
    [data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] p,
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #cbd5e1 !important;
    }

    /* Bold text inside markdown */
    strong, b {
        color: #f1f5f9 !important;
    }

    /* ===================== SLIDER RANGE VALUES ===================== */
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"],
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #94a3b8 !important;
    }

    /* Slider current value text */
    [data-testid="stThumbValue"] {
        color: #38bdf8 !important;
    }

    /* ===================== INPUT FIELDS ===================== */
    /* Text inside input boxes / number inputs / selectboxes */
    input, select, textarea,
    [data-baseweb="input"] input,
    [data-baseweb="select"] div,
    .stSelectbox div[data-baseweb="select"] span,
    .stNumberInput input {
        color: #f1f5f9 !important;
        background-color: #1e293b !important;
        border-color: rgba(148, 163, 184, 0.3) !important;
    }

    /* Selectbox dropdown menu */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #1e293b !important;
    }
    ul[role="listbox"] li {
        color: #e2e8f0 !important;
    }
    ul[role="listbox"] li:hover {
        background-color: #334155 !important;
    }

    /* ===================== BUTTONS ===================== */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 10px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }

    /* ===================== CHECKBOX ===================== */
    .stCheckbox span {
        color: #cbd5e1 !important;
    }

    /* ===================== CONTAINERS WITH BORDER ===================== */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 6px !important;
        backdrop-filter: blur(12px) !important;
    }

    /* ===================== DATAFRAME ===================== */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ===================== ALERTS / INFO / SUCCESS ===================== */
    .stAlert, [data-testid="stNotification"] {
        color: #e2e8f0 !important;
    }

    /* ===================== TABS ===================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
    }

    /* ===================== METRIC CARDS ===================== */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 20px;
    }
    .metric-box {
        flex: 1;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        transition: transform 0.2s ease-in-out, border-color 0.2s;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .metric-box .val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-box .lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* ===================== PREDICTION BADGE ===================== */
    .prediction-badge {
        padding: 10px 24px;
        border-radius: 30px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .setosa {
        background-color: rgba(168, 85, 247, 0.2);
        border: 1px solid rgba(168, 85, 247, 0.5);
        color: #c084fc;
    }
    .versicolor {
        background-color: rgba(249, 115, 22, 0.2);
        border: 1px solid rgba(249, 115, 22, 0.5);
        color: #fdbb2d;
    }
    .virginica {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #34d399;
    }

    /* ===================== SUBTITLE ===================== */
    .subheader-text {
        font-size: 1.15rem;
        color: #94a3b8 !important;
        margin-bottom: 2rem;
    }

    /* ===================== CAPTION ===================== */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }

    /* ===================== DIVIDER ===================== */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🌸 Iris Flower Classification Portal")
st.markdown('<p class="subheader-text">An interactive machine learning environment to analyze, train, evaluate, and predict Iris species using physical flower dimensions.</p>', unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def get_dataset():
    if os.path.exists("Iris.csv"):
        return load_data("Iris.csv")
    else:
        st.error("Dataset 'Iris.csv' not found. Please ensure it has been extracted.")
        return None

df = get_dataset()

if df is not None:
    # Sidebar - Mode Selection
    st.sidebar.markdown("### ⚙️ Workspace Configuration")
    app_mode = st.sidebar.radio(
        "Select App Section",
        ["Dashboard & Analytics", "Interactive Model Trainer", "Real-Time Predictor"]
    )

    # Feature columns list
    feature_cols = [col for col in df.columns if col != 'Species']

    # Emojis and details mapping for species
    species_info = {
        "Iris-setosa": {
            "emoji": "🪻",
            "class_css": "setosa",
            "desc": "Setosa is characterized by small petals and relatively wide sepals. It is highly distinct and easily separable from the other two species.",
            "img_path": "assets/setosa.png"
        },
        "Iris-versicolor": {
            "emoji": "🪷",
            "class_css": "versicolor",
            "desc": "Versicolor lies in the middle range of measurements. It is closely related to Virginica and is often slightly harder to separate linearly.",
            "img_path": "assets/versicolor.png"
        },
        "Iris-virginica": {
            "emoji": "🌸",
            "class_css": "virginica",
            "desc": "Virginica generally features the largest petals and sepals of the three species. It requires robust non-linear boundaries or fine-tuned thresholds to classify correctly.",
            "img_path": "assets/virginica.png"
        }
    }

    # Store trained model in session state so it can be shared between tabs
    if 'trained_model_info' not in st.session_state:
        # Train a default model (Random Forest) so the Predictor works out of the box
        default_hyperparams = {"n_estimators": 100, "max_depth": None}
        model, metrics, X_test, y_test, y_pred, le = train_model(
            df, "Random Forest", default_hyperparams, test_size=0.2, random_state=42
        )
        st.session_state['trained_model_info'] = {
            "model": model,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "le": le,
            "model_type": "Random Forest",
            "hyperparams": default_hyperparams
        }

    # ==========================================
    # SECTION 1: DASHBOARD & ANALYTICS
    # ==========================================
    if app_mode == "Dashboard & Analytics":
        st.markdown("## 📊 Exploratory Data Analysis & Analytics")

        # Row 1: Metrics & Data Preview
        col1, col2 = st.columns([1, 2])

        with col1:
            with st.container(border=True):
                st.markdown("### 🔍 Dataset Summary")
                st.write(f"**Total Samples:** {len(df)}")
                st.write(f"**Features:** {len(feature_cols)} physical dimensions")
                st.write(f"**Classes:** {df['Species'].nunique()} species")

                # Species breakdown
                species_counts = df['Species'].value_counts()
                for spec, cnt in species_counts.items():
                    st.write(f"- **{spec.replace('Iris-', '')}**: {cnt} samples")

                st.markdown("#### Descriptive Statistics")
                st.dataframe(df.describe().T[['mean', 'std', 'min', 'max']], use_container_width=True)

        with col2:
            with st.container(border=True):
                st.markdown("### 📋 Interactive Raw Data View")
                st.dataframe(df, height=330, use_container_width=True)

        # Row 2: Visualizations
        st.markdown("### 📈 Visual Feature Relationships")

        col_vis1, col_vis2 = st.columns(2)

        with col_vis1:
            with st.container(border=True):
                st.markdown("#### 🌸 2D Dimension Clustering")
                x_axis = st.selectbox("X-Axis Feature", feature_cols, index=2)
                y_axis = st.selectbox("Y-Axis Feature", feature_cols, index=3)

                fig = px.scatter(
                    df, x=x_axis, y=y_axis, color="Species",
                    color_discrete_map={
                        "Iris-setosa": "#c084fc",
                        "Iris-versicolor": "#fdbb2d",
                        "Iris-virginica": "#34d399"
                    },
                    labels={x_axis: x_axis.replace("Cm", " (cm)"), y_axis: y_axis.replace("Cm", " (cm)")},
                    title=f"{x_axis} vs {y_axis}"
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_vis2:
            with st.container(border=True):
                st.markdown("#### ⚡ Feature Distributions")
                selected_feat = st.selectbox("Select Feature for Distribution", feature_cols, index=0)

                fig = px.histogram(
                    df, x=selected_feat, color="Species",
                    marginal="box", barmode="overlay",
                    color_discrete_map={
                        "Iris-setosa": "#c084fc",
                        "Iris-versicolor": "#fdbb2d",
                        "Iris-virginica": "#34d399"
                    },
                    labels={selected_feat: selected_feat.replace("Cm", " (cm)")},
                    title=f"Distribution of {selected_feat}"
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

        # Row 3: 3D Scatter & Correlation Heatmap
        col_vis3, col_vis4 = st.columns([3, 2])

        with col_vis3:
            with st.container(border=True):
                st.markdown("#### 🌌 3D Feature Space Visualization")
                z_axis_3d = st.selectbox("3D Z-Axis Feature", feature_cols, index=3)

                fig = px.scatter_3d(
                    df, x='SepalLengthCm', y='PetalLengthCm', z=z_axis_3d,
                    color='Species',
                    color_discrete_map={
                        "Iris-setosa": "#c084fc",
                        "Iris-versicolor": "#fdbb2d",
                        "Iris-virginica": "#34d399"
                    },
                    title="3D Sepal Length vs Petal Length vs Selected Z Feature"
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9'),
                    scene=dict(
                        xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.05)'),
                        zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.05)')
                    ),
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_vis4:
            with st.container(border=True):
                st.markdown("#### 🔗 Correlation Heatmap (Features Only)")

                corr = df[feature_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1.0, zmax=1.0,
                    title="Pearson Correlation Heatmap"
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9')
                )
                st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # SECTION 2: INTERACTIVE MODEL TRAINER
    # ==========================================
    elif app_mode == "Interactive Model Trainer":
        st.markdown("## 🧠 Interactive Machine Learning Laboratory")

        col_ctrl, col_res = st.columns([1, 2])

        with col_ctrl:
            with st.container(border=True):
                st.markdown("### ⚙️ Training Parameters")

                # Select Model Type
                model_selection = st.selectbox(
                    "Classification Algorithm",
                    [
                        "Random Forest",
                        "Support Vector Machine (SVM)",
                        "K-Nearest Neighbors (KNN)",
                        "Logistic Regression",
                        "Decision Tree"
                    ]
                )

                # Test size slider
                test_size = st.slider("Test Dataset Split Fraction", 0.1, 0.5, 0.2, 0.05)

                # Dynamic hyperparameters based on selected model
                st.markdown("#### Hyperparameters")
                hyperparams = {}

                if model_selection == "Random Forest":
                    hyperparams["n_estimators"] = st.number_input("Number of Estimators (Trees)", 10, 500, 100, step=10)
                    max_depth_opt = st.checkbox("Limit Maximum Tree Depth", value=False)
                    hyperparams["max_depth"] = st.slider("Max Depth", 1, 30, 5) if max_depth_opt else None

                elif model_selection == "Support Vector Machine (SVM)":
                    hyperparams["C"] = st.number_input("Regularization Parameter (C)", 0.01, 100.0, 1.0, step=0.5)
                    hyperparams["kernel"] = st.selectbox("Kernel Function", ["rbf", "linear", "poly", "sigmoid"])

                elif model_selection == "K-Nearest Neighbors (KNN)":
                    hyperparams["n_neighbors"] = st.slider("Number of Neighbors (K)", 1, 25, 5)
                    hyperparams["weights"] = st.selectbox("Weight Function", ["uniform", "distance"])

                elif model_selection == "Logistic Regression":
                    hyperparams["C"] = st.number_input("Inverse Regularization Strength (C)", 0.01, 100.0, 1.0, step=0.5)
                    hyperparams["max_iter"] = st.number_input("Max Iterations", 50, 1000, 200, step=50)

                elif model_selection == "Decision Tree":
                    max_depth_opt = st.checkbox("Limit Maximum Tree Depth", value=False)
                    hyperparams["max_depth"] = st.slider("Max Depth", 1, 30, 5) if max_depth_opt else None

                # Random seed for consistency
                random_state = st.number_input("Random Seed (Reproducibility)", 0, 1000, 42)

                # Trigger Training
                train_trigger = st.button("🔥 Run Training", use_container_width=True)

        with col_res:
            if train_trigger:
                with st.spinner(f"Training {model_selection}..."):
                    # Train model and update session state
                    model, metrics, X_test, y_test, y_pred, le = train_model(
                        df, model_selection, hyperparams, test_size, random_state
                    )

                    st.session_state['trained_model_info'] = {
                        "model": model,
                        "metrics": metrics,
                        "X_test": X_test,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "le": le,
                        "model_type": model_selection,
                        "hyperparams": hyperparams
                    }
                    st.success(f"Successfully trained {model_selection}!")

            # Retrieve model information from session state
            model_info = st.session_state['trained_model_info']

            with st.container(border=True):
                st.markdown(f"### 📊 Model Evaluation: `{model_info['model_type']}`")

                # Visual metrics boxes
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-box">
                        <div class="val">{model_info['metrics']['accuracy'] * 100:.1f}%</div>
                        <div class="lbl">Test Accuracy</div>
                    </div>
                    <div class="metric-box">
                        <div class="val">{model_info['metrics']['f1_score'] * 100:.1f}%</div>
                        <div class="lbl">Weighted F1-Score</div>
                    </div>
                    <div class="metric-box">
                        <div class="val">{model_info['metrics']['precision'] * 100:.1f}%</div>
                        <div class="lbl">Weighted Precision</div>
                    </div>
                    <div class="metric-box">
                        <div class="val">{model_info['metrics']['recall'] * 100:.1f}%</div>
                        <div class="lbl">Weighted Recall</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_eval1, col_eval2 = st.columns(2)

                with col_eval1:
                    st.markdown("#### 🎯 Confusion Matrix")
                    # Confusion Matrix Heatmap
                    cm = confusion_matrix(model_info['y_test'], model_info['y_pred'])
                    classes_labels = [c.replace('Iris-', '') for c in model_info['le'].classes_]

                    fig = px.imshow(
                        cm,
                        x=classes_labels,
                        y=classes_labels,
                        text_auto=True,
                        color_continuous_scale="BuPu",
                        labels=dict(x="Predicted Species", y="Actual Species"),
                        title="Test Confusion Matrix"
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#f1f5f9'),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_eval2:
                    st.markdown("#### 📈 Model Insights")
                    # Feature importance or coefficients
                    model = model_info['model']
                    m_type = model_info['model_type']

                    has_importance = False
                    importance_vals = []

                    if m_type == "Random Forest" or m_type == "Decision Tree":
                        importance_vals = model.feature_importances_
                        has_importance = True
                    elif m_type == "Logistic Regression":
                        # Take average absolute weight coefficients across classes
                        importance_vals = np.mean(np.abs(model.coef_), axis=0)
                        has_importance = True
                    elif m_type == "Support Vector Machine (SVM)" and model_info['hyperparams'].get("kernel") == "linear":
                        importance_vals = np.mean(np.abs(model.coef_.toarray() if hasattr(model.coef_, "toarray") else model.coef_), axis=0)
                        has_importance = True

                    if has_importance:
                        feat_imp_df = pd.DataFrame({
                            "Feature": [c.replace('Cm', ' (cm)') for c in feature_cols],
                            "Importance": importance_vals
                        }).sort_values("Importance", ascending=True)

                        fig = px.bar(
                            feat_imp_df,
                            y="Feature",
                            x="Importance",
                            orientation="h",
                            title="Relative Feature Importance",
                            color_discrete_sequence=["#38bdf8"]
                        )
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f1f5f9'),
                            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"Feature importance visualization is not directly applicable or available for {m_type} (kernel: {model_info['hyperparams'].get('kernel', 'N/A')}).")

    # ==========================================
    # SECTION 3: REAL-TIME PREDICTOR
    # ==========================================
    elif app_mode == "Real-Time Predictor":
        st.markdown("## 🔮 Real-Time Species Inference")

        # Load the model from session state
        model_info = st.session_state['trained_model_info']
        model = model_info['model']
        le = model_info['le']

        col_pred_in, col_pred_out = st.columns([1, 1])

        with col_pred_in:
            with st.container(border=True):
                st.markdown("### 📏 Input Physical Measurements (cm)")

                # Form inputs based on min/max in training dataset
                sepal_length = st.slider(
                    "Sepal Length (cm)",
                    float(df['SepalLengthCm'].min()),
                    float(df['SepalLengthCm'].max()),
                    float(df['SepalLengthCm'].mean()),
                    0.1
                )
                sepal_width = st.slider(
                    "Sepal Width (cm)",
                    float(df['SepalWidthCm'].min()),
                    float(df['SepalWidthCm'].max()),
                    float(df['SepalWidthCm'].mean()),
                    0.1
                )
                petal_length = st.slider(
                    "Petal Length (cm)",
                    float(df['PetalLengthCm'].min()),
                    float(df['PetalLengthCm'].max()),
                    float(df['PetalLengthCm'].mean()),
                    0.1
                )
                petal_width = st.slider(
                    "Petal Width (cm)",
                    float(df['PetalWidthCm'].min()),
                    float(df['PetalWidthCm'].max()),
                    float(df['PetalWidthCm'].mean()),
                    0.1
                )

                # Predict values
                input_df = pd.DataFrame([[
                    sepal_length, sepal_width, petal_length, petal_width
                ]], columns=feature_cols)

                pred_label, probs, classes = predict_species(model, le, input_df)

        with col_pred_out:
            with st.container(border=True):
                st.markdown("### 🎯 Classification Results")

                info = species_info.get(pred_label, {"emoji": "🌸", "class_css": "setosa", "desc": "Unknown species", "img_path": ""})

                # Display Prediction Badge
                st.markdown("**Predicted Species:**")
                st.markdown(f'<span class="prediction-badge {info["class_css"]}">{info["emoji"]} {pred_label.replace("Iris-", "")}</span>', unsafe_allow_html=True)

                st.markdown(f"<p style='margin-top:15px; color:#94a3b8;'>{info['desc']}</p>", unsafe_allow_html=True)

                # Display Probability Distribution
                if probs is not None:
                    st.markdown("#### Probability Distribution:")
                    prob_df = pd.DataFrame({
                        "Species": [c.replace('Iris-', '') for c in classes],
                        "Probability": probs
                    })

                    # Highlight predicted class in chart
                    fig = px.bar(
                        prob_df, x="Probability", y="Species",
                        orientation="h",
                        color="Species",
                        color_discrete_map={
                            "setosa": "#c084fc",
                            "versicolor": "#fdbb2d",
                            "virginica": "#34d399"
                        },
                        range_x=[0.0, 1.05]
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#f1f5f9'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickformat=".0%"),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                        showlegend=False,
                        height=200,
                        margin=dict(l=0, r=0, b=0, t=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # If the image asset exists, show it!
                if info["img_path"] and os.path.exists(info["img_path"]):
                    try:
                        img = Image.open(info["img_path"])
                        st.image(img, caption=f"Specimen: {pred_label.replace('Iris-', '')}", use_container_width=True)
                    except Exception as e:
                        pass
                else:
                    st.caption("*(Optional: Generate or place species images in `assets/` to display specimen previews)*")

else:
    st.info("Loading dataset. Please ensure 'Iris.csv' is in the root directory.")
