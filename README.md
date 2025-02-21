# DataVivid ML App 🚀

## Demo
https://github.com/user-attachments/assets/bfd0941a-5cbf-4666-b9e6-ff2bd1e46f58
## Overview
This repository hosts *DataVivid ML App*, a dynamic Streamlit-based tool designed to simplify data exploration and machine learning. Upload a CSV file, dive into colorful visualizations, preprocess your data with ease, and unleash powerful ML models—all with an intuitive, flair-filled interface. Built with Python, Scikit-learn, and a passion for vivid data insights, this app empowers users of all levels to analyze and predict with style! 🌟

## How It Works
The app’s workflow is designed to be engaging and user-friendly, guiding you from data upload to actionable insights:

1. **User Input**: Upload a CSV file to kick things off 📂.
2. **Data Exploration**: Preview your data, view descriptive stats, and analyze value counts with adjustable sliders and tables 📊.
3. **Visualization Generation**: Choose from vibrant Heatmaps 🔥, Pairplots 🌐, Violinplots 🎻, or Scatterplots 📉, with customizable feature selection.
4. **Preprocessing**: Drop columns 🗑️ or fill missing values using Mean, Median, Mode, or Zero—tailored for numeric and categorical data 🛠️.
5. **Model Training and Prediction**:
   - Select Classification or Regression, pick a model (e.g., Random Forest, Neural Network), and choose a target 🎯.
   - Adjust test split size and optionally standardize features 📏.
   - Hit "Predict" to train the model and view performance metrics and feature importance 🌟.
6. **Data Visualization and Insights**: Explore results with bar charts for feature importance or coefficients, alongside detailed metrics 📈.

## Enhanced Capabilities with ML
By leveraging Scikit-learn’s robust ML algorithms, *DataVivid ML App* brings cutting-edge modeling to your fingertips. From Random Forests to Neural Networks, the app auto-detects task types (classification vs. regression) based on your target, delivering precise predictions and insights. The colorful visualizations and interactive controls make complex ML accessible and fun! 🤖

## Challenges Encountered
Building this app came with its share of hurdles:

- **Missing Value Handling**: Ensuring seamless preprocessing for both categorical and numeric data required careful logic to avoid errors while filling NaNs.
- **Visualization Performance**: Pairplots with large datasets or many features could slow down; the app mitigates this with warnings and selective feature options.
- **Streamlit Layout**: Managing the sequential rendering of widgets in Streamlit demanded a thoughtful UI design to keep the experience smooth and logical.

Please, try the app here:  
[Streamlit App](https://datavivid-ml.streamlit.app/)

---

### Installation ⚙️
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/datavivid-ml-app.git
   cd datavivid-ml-app

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
requirements.txt:
   streamlit
   pandas
   seaborn
   matplotlib
   scikit-learn

3.Run the app:
   ```bash
   streamlit run app.py
   ```
The app will launch locally. Visit http://localhost:8501 in your browser to dive into vibrant data exploration and ML modeling! 🎉

## Usage 📝
- Upload a CSV file and explore it with the sidebar tools.
- Visualize your data with flair-filled plots.
- Train an ML model and interpret results with vibrant charts and metrics.
- Enjoy the process—data analysis has never been this vivid! 🌈

## Contributing 🤝
Love data and vibrant visuals? Fork this repo, report issues, or submit pull requests to enhance *DataVivid ML App*. Let’s make data exploration dazzling together!

Built with 💖 by piyushk1304 using Streamlit and Scikit-learn. Dive in and make your data pop! 🎉
