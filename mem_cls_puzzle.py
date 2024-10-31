import nltk

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import argparse
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os 
import json

def preprocess_text(text):
    try:
        # If text is a list, concatenate all elements into a single string
        if isinstance(text, list):
            text = " ".join(text)

        # Lowercase and remove punctuation
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""


def vectorize_text(train, test, text_field="quiz", method="tfidf", num_ppl=5):
    column_names = train.columns.tolist()
    if f"clean_{text_field}" in column_names:
        text_field = f"clean_{text_field}" # use clean data's field (not perturbed data)
    
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=5000)
        train_feature = vectorizer.fit_transform(train["processed_text"])
        test_feature = vectorizer.transform(test["processed_text"])
        train_feature = train_feature.toarray()
        test_feature = test_feature.toarray()
    elif method == "bow":
        vectorizer = CountVectorizer(max_features=5000)
        train_feature = vectorizer.fit_transform(train["processed_text"])
        test_feature = vectorizer.transform(test["processed_text"])
        train_feature = train_feature.toarray()
        test_feature = test_feature.toarray()

    elif method == "charlength":
        train_feature = np.asarray(
            [len(s) for s in train["processed_text"].values]
        ).reshape(-1, 1)
        test_feature = np.asarray(
            [len(s) for s in test["processed_text"].values]
        ).reshape(-1, 1)
    elif method == "wordlength":
        train_feature = np.asarray(
            [len(s.split(" ")) for s in train["processed_text"].values]
        ).reshape(-1, 1)
        test_feature = np.asarray(
            [len(s.split(" ")) for s in test["processed_text"].values]
        ).reshape(-1, 1)
    


    return train_feature, test_feature


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run classification for memorization.")
    parser.add_argument(
        "--train_split", type=float, default=0.8, help="Fraction for training"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tfidf", "bow", "wordlength", "charlength",  "combine",],
        default="charlength",
        help="Vectorization method",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        choices=[
            "quiz",
            "names",
            "solution",
            "solution_text",
            "solution_text_format",
            "cot_steps",
            "cot_repeat_steps",
            "statements",
            "response",
            "all_fields",
            "state_quiz",
            "state_quiz_resp",
            "quiz_resp",
            "state_resp",
        ],
        default="quiz",
        help="The field to featurize",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="",
        help="Path to data jsonl file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="result/", help="Directory to save output CSV"
    )
    parser.add_argument("--no_balance_label", action="store_true")

    return parser.parse_args()


def prepare_cls_data(df, train_split=0.8):
    return train_test_split(
        df,
        test_size=1 - train_split,
        stratify=df["label"],
        random_state=42,
    )

def train_and_evaluate(train_feature, test_feature, train_label, test_label):
    model = LogisticRegression(random_state=42,max_iter=10000)
    model.fit(train_feature, train_label)

    train_pred = model.predict(train_feature)
    test_pred = model.predict(test_feature)

    # Predict probabilities instead of labels
    train_probs = model.predict_proba(train_feature)
    test_probs = model.predict_proba(test_feature)
    
    
    evaluation= {
        "train_accuracy": accuracy_score(train_label, train_pred),
        "test_accuracy": accuracy_score(test_label, test_pred),
        "train_auc": roc_auc_score(train_label, train_probs[:, 1]),
        "test_auc":roc_auc_score(test_label, test_probs[:, 1]),
        
    }
    report= classification_report(test_label, test_pred,output_dict=True)
    evaluation.update(report)
    return evaluation


def main():
    args = parse_arguments()
    data = pd.read_json(args.input_file, lines=True)
    data["label"] = data["robust_metric"]
    num_ppl= int(args.input_file.split("/")[-1].split("_")[0].replace("people",""))
    print(num_ppl)

    if args.no_balance_label==False:
        # Separate the data by label
        data_0 = data[data["label"] == 0]
        data_1 = data[data["label"] == 1]

        # Determine the size of the smaller class
        min_size = min(len(data_0), len(data_1))

        # Sample from each class to balance the dataset
        balanced_data_0 = data_0.sample(n=min_size, random_state=42)
        balanced_data_1 = data_1.sample(n=min_size, random_state=42)

        # Concatenate the balanced datasets
        balanced_data = pd.concat([balanced_data_0, balanced_data_1])

        # Shuffle the balanced dataset
        data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    train, test = prepare_cls_data(data, args.train_split)
    
    
    methods=[]
    if args.method=="combine":   
        methods=["tfidf",  "bow", "wordlength" , "charlength",]
  
    else:
        methods=[args.method]

    train_feature_list=[]
    test_feature_list=[] 
    if args.text_field =="all_fields":  
        text_fields = ["statements", "quiz" ,"response" ,   "cot_repeat_steps", "cot_steps", ] 
    else:
        text_fields = [args.text_field]   
    
    for text_field in text_fields:
        for method in methods:   
           
            print(f"Processing {text_field} with {method}")

            train_feature, test_feature = vectorize_text(
                train, test, text_field=text_field, method=method, num_ppl=num_ppl
            )
            train_feature_list.append(train_feature)
            test_feature_list.append(test_feature)

    # Initialize an empty array
    concatenated_features = train_feature_list[0]
    print(len(train_feature_list))
    # Use a for loop to concatenate the features
    if len(train_feature_list)>1:
        for i, feature in enumerate(train_feature_list[1:]):
            concatenated_features = np.concatenate((concatenated_features, feature), axis=1)
    train_feature=concatenated_features


    print(len(test_feature_list))
    concatenated_features = test_feature_list[0]
    if len(test_feature_list)>1:
        for feature in test_feature_list[1:]:
            concatenated_features = np.concatenate((concatenated_features, feature), axis=1)
    test_feature=concatenated_features
    

    print("Train_feature shape", train_feature.shape)
    print("Test_feature shape", test_feature.shape)

   
    evaluation={}
    evaluation["method"] = args.method
    evaluation["text_field"] = args.text_field
    evaluation["input_file"] = args.input_file
    evaluation_results = train_and_evaluate(
        train_feature,
        test_feature,
        train["label"],
        test["label"],
    )
    evaluation.update(evaluation_results)

    print(evaluation)
   
    # # TODO: save eval results
    os.makedirs(args.output_dir, exist_ok=True)
    if args.no_balance_label:
        output_file = os.path.join(args.output_dir, f"results_{num_ppl}_unbalanced.jsonl")
    else:
        output_file = os.path.join(args.output_dir, f"results_{num_ppl}_balanced.jonsl")
    # Read existing data
    existing_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            for line in file:
                existing_data.append(json.loads(line))
    existing_data.append(evaluation)

    # Write all data back to the file
    with open(output_file, 'w') as file:
        for item in existing_data:
            json.dump(item, file)
            file.write('\n')



if __name__ == "__main__":
    main()
