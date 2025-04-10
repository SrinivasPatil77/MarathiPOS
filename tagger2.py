import streamlit as st
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import re

# --------------------- DATA HANDLING ---------------------

def read_ssf_file(filepath):
    sentences = []
    current_sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("<Corpora"):
                continue
            if line.startswith("<Sentence"):
                current_sentence = []
            elif line.startswith("</Sentence>"):
                if current_sentence:
                    sentences.append(current_sentence)
            elif '\t' in line:
                parts = line.split('\t')
                if len(parts) == 3:
                    _, word, tag = parts
                    current_sentence.append((word, tag))
        if current_sentence:
            sentences.append(current_sentence)
    
    return sentences

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word': word,
        'prefix3': word[:3],
        'suffix3': word[-3:],
        'is_digit': word.isdigit(),
        'is_upper': word.isupper(),
    }
    if i > 0:
        features['-1:word'] = sent[i-1][0]
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        features['+1:word'] = sent[i+1][0]
    else:
        features['EOS'] = True

    return features

def extract_features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def get_labels(sent):
    return [label for token, label in sent]

def prepare_data(data):
    X = [extract_features(s) for s in data]
    y = [get_labels(s) for s in data]
    return X, y

def tokenize_marathi(text):
    return text.strip().split()

# --------------------- TRAINING ---------------------

@st.cache_resource
def train_crf_model(filepath):
    data = read_ssf_file(filepath)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    accuracy = metrics.flat_accuracy_score(y_test, crf.predict(X_test))
    return crf, accuracy

# --------------------- STREAMLIT UI ---------------------

def main():
    st.set_page_config(page_title="Marathi PoS Tagger", layout="wide")
    st.title("📚 Marathi PoS Tagger using CRF")

    filepath = "test.utf.ssf.pos"
    with st.spinner("Training CRF model..."):
        crf_model, accuracy = train_crf_model(filepath)

    st.success(f"Model trained with Accuracy: {accuracy:.2%}")

    user_input = st.text_input("✍️ Enter a Marathi sentence:")
    
    if user_input:
        tokens = tokenize_marathi(user_input)
        dummy_sent = [(w, 'X') for w in tokens]  # dummy tags for feature extraction
        features = extract_features(dummy_sent)
        prediction = crf_model.predict([features])[0]

        st.markdown("### 🔍 PoS Tags:")
        for word, tag in zip(tokens, prediction):
            st.write(f"**{word}** → _{tag}_")

if __name__ == "__main__":
    main()
