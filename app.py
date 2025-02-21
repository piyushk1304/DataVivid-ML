import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
st.set_page_config(
    page_title="DataVivid ML App",      
    page_icon=":rocket:",               
)
@st.cache_data
def readcsv(csv):
    df = pd.read_csv(csv)
    return df

def head(dataframe):
    if len(dataframe) > 1000:
        length = 1000
    else:
        length = len(dataframe)
    slider = st.slider('Number of rows displayed 📏', 5, length)
    st.dataframe(dataframe.head(slider))

@st.cache_data
def exploratory_func(dataframe):
    desc = dataframe.describe().T
    desc['column'] = desc.index
    exploratory = pd.DataFrame()
    exploratory['NaN'] = dataframe.isnull().sum().values
    exploratory['NaN %'] = 100 * (dataframe.isnull().sum().values / len(dataframe))
    exploratory['NaN %'] = exploratory['NaN %'].apply(lambda x: f"{round(x, 2)} %")
    exploratory['column'] = dataframe.columns
    exploratory['dtype'] = dataframe.dtypes.values
    exploratory = exploratory.merge(desc, on='column', how='left')
    exploratory.loc[exploratory['dtype'] == 'object', 'count'] = len(dataframe) - exploratory['NaN']
    exploratory.set_index('column', inplace=True)
    return exploratory

def heatmap(dataframe, cols=[]):
    numcols = tuple(dataframe.select_dtypes(exclude='object').columns)
    check = st.checkbox('Select all ✅', key='heatmap')
    if check:
        cols = list(numcols)
    select = st.multiselect('Numeric features 📊:', numcols, default=cols, key='heatmap_select')
    if len(select) > 10:
        annot = False
    else:
        annot = True
    if len(select) > 1:
        fig, ax = plt.subplots()
        # Use a vibrant colormap
        sns.heatmap(dataframe[select].corr(), annot=annot, cmap='coolwarm', ax=ax, fmt='.2f', 
                    cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Heatmap', fontsize=14, pad=10)
        st.pyplot(fig)

def pairplot(dataframe, cols=[]):
    catcols = ['-'] + list(dataframe.select_dtypes(include='object').columns)
    numcols = tuple(dataframe.select_dtypes(exclude='object').columns)
    
    if not numcols:
        st.warning("No numeric columns available for pairplot. ⚠️")
        return
    
    check = st.checkbox('Select all ✅', key='pairplot')
    if check:
        cols = list(numcols)
    select = st.multiselect('Numeric features 📈:', numcols, default=cols, key='pairplot_select')
    hue = st.selectbox('Select hue 🎨', catcols, key='pairplot_hue')

    if len(select) < 2:
        st.warning("Please select at least 2 numeric features for the pairplot. ⚠️")
        return
    
    try:
        # Use a colorful palette
        if hue == '-':
            g = sns.pairplot(dataframe[select], diag_kind='kde', palette='husl', 
                             plot_kws={'alpha': 0.6, 's': 50}, diag_kws={'color': 'purple'})
        else:
            copy = select + [hue]
            g = sns.pairplot(dataframe[copy], hue=hue, palette='husl', diag_kind='kde', 
                             plot_kws={'alpha': 0.6, 's': 50}, diag_kws={'color': 'purple'})
        g.fig.suptitle('Pairplot of Numeric Features', y=1.02, fontsize=14)
        st.pyplot(g.figure)
        plt.close(g.figure)
    except Exception as e:
        st.error(f"Error generating pairplot: {str(e)} ❌")

def boxplot(dataframe):
    numcol = tuple(dataframe.select_dtypes(exclude='object').columns)
    catcol = tuple(dataframe.select_dtypes(include='object').columns)
    select1 = st.selectbox('Select a numeric feature 📊', numcol, key='boxplot_num')
    select2 = st.selectbox('Select a categorical feature 🗂️', catcol, key='boxplot_cat')
    fig, ax = plt.subplots()
    # Use a colorful palette for violinplot
    sns.violinplot(data=dataframe, x=select2, y=select1, ax=ax, palette='Set2', inner='quartile')
    ax.set_title(f'Violinplot: {select2} vs {select1}', fontsize=14, pad=10)
    st.pyplot(fig)

def scatter(dataframe):
    numcol = tuple(dataframe.select_dtypes(exclude='object').columns)
    select1 = st.selectbox('Select numeric feature (X) 📏', numcol, key='scatter_x')
    select2 = st.selectbox('Select another numeric feature (Y) 📏', numcol, key='scatter_y')
    fig, ax = plt.subplots()
    # Add color and size variation
    ax.scatter(dataframe[select1], dataframe[select2], c='teal', alpha=0.6, s=100, edgecolors='black')
    ax.set_xlabel(select1, fontsize=12)
    ax.set_ylabel(select2, fontsize=12)
    ax.set_title(f'Scatter: {select1} vs {select2}', fontsize=14, pad=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def valuecounts(dataframe):
    select = st.selectbox('Select one feature 📋:', tuple(dataframe.columns), key='valuecounts')
    st.write(dataframe[select].value_counts())

def input(dataframe, nan_input, features):
    for feature in features:
        if feature not in dataframe.columns:
            continue
        if dataframe[feature].dtype in ['object', 'category']:
            if nan_input in ['Mean', 'Median']:
                st.warning(f"Skipping {feature}: Mean/Median not applicable to categorical data. ⚠️")
                continue
            elif nan_input == 'Mode':
                fill_value = dataframe[feature].mode()[0] if not dataframe[feature].mode().empty else 'Unknown'
                dataframe[feature].fillna(fill_value, inplace=True)
            elif nan_input == 'Zero':
                dataframe[feature].fillna('0', inplace=True)
        else:
            if nan_input == 'Mean':
                fill_value = dataframe[feature].mean() if not dataframe[feature].isna().all() else 0
                dataframe[feature].fillna(fill_value, inplace=True)
            elif nan_input == 'Median':
                fill_value = dataframe[feature].median() if not dataframe[feature].isna().all() else 0
                dataframe[feature].fillna(fill_value, inplace=True)
            elif nan_input == 'Mode':
                fill_value = dataframe[feature].mode()[0] if not dataframe[feature].mode().empty else 0
                dataframe[feature].fillna(fill_value, inplace=True)
            elif nan_input == 'Zero':
                dataframe[feature].fillna(0, inplace=True)

def drop(dataframe, select):
    if len(select) != 0:
        return dataframe.drop(columns=select)
    return dataframe

def is_continuous(y):
    if pd.api.types.is_numeric_dtype(y):
        unique_ratio = len(y.unique()) / len(y)
        return unique_ratio > 0.05
    return False

def main():
    st.title('DataVivid ML 🚀')
    st.markdown("Drop in a CSV 📂, tweak and explore with flair 🎨 via the sidebar, and unleash vibrant predictions 🌟 with top ML models 🤖!")
    
    file = st.file_uploader('Upload your CSV file 📂', type='csv')
    if file is not None:
        st.sidebar.subheader('Visualization 🎨')
        sidemulti = st.sidebar.multiselect('Plots:', ('Heatmap', 'Pairplot', 'Violinplots', 'Scatterplot'))

        df0 = pd.DataFrame(readcsv(file))
        st.sidebar.subheader('Drop columns 🗑️:')
        sidedrop = st.sidebar.multiselect('Columns to be dropped:', tuple(df0.columns))
        df = drop(df0, sidedrop)
        
        st.sidebar.subheader('Fill missing values in features 🛠️')
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            st.sidebar.write(f"Columns with missing values: {', '.join(missing_cols)}")
            sidemethod = st.sidebar.selectbox('Method to fill all missing values:', 
                                            ('Mean', 'Median', 'Mode', 'Zero'), 
                                            key='fill_method')
            if st.sidebar.button('Fill Missing Values ✅'):
                input(df, sidemethod, missing_cols)
                st.sidebar.success(f"Filled missing values in {', '.join(missing_cols)} using {sidemethod} 🎉")
        else:
            st.sidebar.write("No missing values detected in features. ✅")

        st.header('Dataframe Visualization 📊')
        head(df)
        st.header('Descriptive Statistics 📈')
        st.dataframe(exploratory_func(df))
        st.header('Value Counts 🔢')
        valuecounts(df)

        if 'Heatmap' in sidemulti:
            st.header('Heatmap 🔥')
            st.subheader('Select numeric features:')
            heatmap(df)

        if 'Pairplot' in sidemulti:
            st.header('Pairplot 🌐')
            st.subheader('Select numeric features and 1 categorical feature at most')
            pairplot(df)

        if 'Violinplots' in sidemulti:
            st.header('Select features for the Violinplot 🎻')
            boxplot(df)
        
        if 'Scatterplot' in sidemulti:
            st.header('Select features for the Scatterplot 📉')
            scatter(df)

        st.header('Machine Learning Models 🤖:')
        model_options = {
            'Classification': [
                'Random Forest Classifier',
                'Logistic Regression',
                'Support Vector Classifier',
                'Gradient Boosting Classifier',
                'Neural Network Classifier'
            ],
            'Regression': [
                'Random Forest Regressor',
                'Linear Regression',
                'Support Vector Regressor',
                'Gradient Boosting Regressor',
                'Neural Network Regressor'
            ]
        }
        
        task_type = st.selectbox('Select task type:', ('Classification', 'Regression'), key='task')
        model = st.selectbox('Select model 🧠:', model_options[task_type], key='model')
        target = st.selectbox('Select the target column 🎯:', tuple(df.columns), key='target')
        
        y = df[target]
        if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
            st.error("Please select a single target column. Multi-output targets are not supported for feature importance visualization. ❌")
            return
        
        if df.isnull().any().any():
            default_method = 'Mean'
            missing_cols = df.columns[df.isnull().any()].tolist()
            st.info(f"Automatically filling {df.isnull().sum().sum()} missing values in {missing_cols} with {default_method} to proceed. ℹ️")
            input(df, default_method, missing_cols)
        
        X = pd.get_dummies(df.drop(columns=target))
        y = y if isinstance(y, pd.Series) else y.iloc[:, 0]

        if X.isnull().any().any() or y.isnull().any():
            if X.isnull().any().any():
                st.error(f"Feature matrix X still contains NaN values in columns: {X.columns[X.isnull().any()].tolist()} ❌")
            if y.isnull().any():
                st.error(f"Target y still contains {y.isnull().sum()} NaN values ❌")
            st.error("Preprocessing failed to remove all NaN values. Please check your data and try again. ⚠️")
            return

        tt_slider = st.slider('% Size of test split ⚖️:', 1, 99, value=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01 * tt_slider, random_state=42)

        scale_data = st.checkbox('Standardize features (recommended for SVM and Neural Networks) 📏', value=False)

        predict = st.button('Predict 🚀')
        if predict:
            st.header('Results: 🎉')
            st.subheader('Performance Metrics 📊:')
            try:
                continuous_target = is_continuous(y)
                
                if model == 'Random Forest Classifier':
                    if continuous_target:
                        st.error("Target is continuous but a Classifier was selected. ❌")
                        return
                    clf = RandomForestClassifier(random_state=42)
                elif model == 'Random Forest Regressor':
                    clf = RandomForestRegressor(random_state=42)
                elif model == 'Logistic Regression':
                    if continuous_target:
                        st.error("Target is continuous but a Classifier was selected. ❌")
                        return
                    clf = LogisticRegression(random_state=42, max_iter=1000)
                elif model == 'Linear Regression':
                    clf = LinearRegression()
                elif model == 'Support Vector Classifier':
                    if continuous_target:
                        st.error("Target is continuous but a Classifier was selected. ❌")
                        return
                    clf = SVC(random_state=42)
                elif model == 'Support Vector Regressor':
                    clf = SVR()
                elif model == 'Gradient Boosting Classifier':
                    if continuous_target:
                        st.error("Target is continuous but a Classifier was selected. ❌")
                        return
                    clf = GradientBoostingClassifier(random_state=42)
                elif model == 'Gradient Boosting Regressor':
                    clf = GradientBoostingRegressor(random_state=42)
                elif model == 'Neural Network Classifier':
                    if continuous_target:
                        st.error("Target is continuous but a Classifier was selected. ❌")
                        return
                    clf = MLPClassifier(random_state=42, max_iter=1000)
                elif model == 'Neural Network Regressor':
                    clf = MLPRegressor(random_state=42, max_iter=1000)

                if scale_data:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                if task_type == 'Classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    st.markdown(f"Accuracy: {accuracy:.4f} ✅")
                    st.markdown(f"Precision (weighted): {precision:.4f} 🎯")
                    st.markdown(f"Recall (weighted): {recall:.4f} 📞")
                    st.markdown(f"F1-Score (weighted): {f1:.4f} ⚖️")
                else:
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    st.markdown(f"R² Score: {r2:.4f} 📈")
                    st.markdown(f"Mean Squared Error: {mse:.4f} 📉")
                    st.markdown(f"Mean Absolute Error: {mae:.4f} ⚖️")
                
                if hasattr(clf, 'feature_importances_'):
                    st.subheader('Feature Importances 🌟:')
                    feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    feat_importances.nlargest(10).plot(kind='barh', ax=ax, color='skyblue', edgecolor='black')
                    ax.set_title('Top 10 Feature Importances', fontsize=14, pad=10)
                    ax.set_xlabel('Importance', fontsize=12)
                    st.pyplot(fig)
                elif hasattr(clf, 'coef_'):
                    st.subheader('Feature Coefficients 📊:')
                    coef = clf.coef_
                    if coef.ndim > 1:
                        st.warning("Multi-output target detected. Showing coefficients for the first output only. ⚠️")
                        coef = coef[0]
                    feat_importances = pd.Series(coef, index=X.columns)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    feat_importances.nlargest(10).abs().plot(kind='barh', ax=ax, color='salmon', edgecolor='black')
                    ax.set_title('Top 10 Feature Coefficients', fontsize=14, pad=10)
                    ax.set_xlabel('Coefficient (Absolute)', fontsize=12)
                    st.pyplot(fig)

            except ValueError as e:
                st.error(f"Error: {str(e)}. Please ensure the target matches the model type and is a single column. ❌")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}. Check data consistency and try again. ⚠️")

if __name__ == '__main__':
    main()