# HealthcarePredict

### Project Overview
This project focuses on building a text classification model to classify medical questions based on their "focus area" using a Bag of Words (BoW) representation of the question text. We leverage a neural network architecture based on Long Short-Term Memory (LSTM) to process the sequential data and make predictions. The dataset used is from Kaggle and contains medical questions with their corresponding answers and focus areas.

### Libraries Used
- **TensorFlow/Keras**: For building and training the neural network.
- **scikit-learn**: For data preprocessing and model evaluation.
- **NLTK**: For text cleaning and stopword removal.
- **KaggleHub**: For dataset download.
- **Matplotlib**: For visualization.
- **NumPy and Pandas**: For data manipulation.

### Dataset
The dataset contains questions related to medical topics and their corresponding answers and focus areas. The dataset is cleaned, and the questions are preprocessed to remove stopwords and special characters.

### Steps Taken:
1. **Data Loading**: The dataset is downloaded using KaggleHub and loaded into a pandas DataFrame.
2. **Data Preprocessing**:
   - Missing values were handled by removing rows with null values.
   - Text data was cleaned by converting to lowercase, removing special characters, and eliminating stopwords.
3. **Text Representation**: Bag of Words (BoW) was used to convert questions into numerical features based on word frequency.
4. **Label Encoding**: The focus area labels were encoded using LabelEncoder to convert categorical values into numerical values.
5. **Model Building**:
   - A simple LSTM-based neural network was used to process the sequence of words in the questions.
   - Layers like Dropout, GlobalMaxPooling, and Dense layers were employed to reduce overfitting and improve generalization.
6. **Model Training**: The model was trained using sparse categorical cross-entropy loss and Adam optimizer with early stopping to prevent overfitting.
7. **Evaluation**: The model's performance was evaluated on a test dataset.
