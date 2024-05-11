from transformers import pipeline
from background_and_models import *

train_path = "./multinli_1.0/multinli_1.0_train.jsonl"
train_data = load_data(train_path)

dev_matched_path = "./multinli_1.0/multinli_1.0_dev_matched.jsonl"
dev_matched = load_data(dev_matched_path)

grammatical_path = "./multinli_1.0/grammatical_dataset.json"
negated_grammatical_path = "./multinli_1.0/negated_grammatical_dataset.json"

if os.path.exists(grammatical_path):
        with open(grammatical_path, "r") as json_file:
            loaded_data = json.load(json_file)
        grammatical_train = [Datum.from_dict(data) for data in loaded_data]

# load negated grammatical_data if it exists
if os.path.exists(negated_grammatical_path):
    with open(grammatical_path, "r") as json_file:
        loaded_data = json.load(json_file)
    augmented_train = [Datum.from_dict(data) for data in loaded_data]
    print(f"NEGATED DATA FOUND!")
    print(f"There are {len(augmented_train) } grammatical sentences and negations in the augmented training data.")

# if it doesn't exist, check for grammatical data, then negate
else:
    if os.path.exists(grammatical_path):
        with open(grammatical_path, "r") as json_file:
            loaded_data = json.load(json_file)
        grammatical_train = [Datum.from_dict(data) for data in loaded_data]

        neg_train_data = create_negated(grammatical_train)
        augmented_train = train_data + neg_train_data
        random.shuffle(augmented_train)

        serialized_data = [datum.to_dict() for datum in augmented_train]

        with open(negated_grammatical_path, "w") as json_file:
            json.dump(serialized_data, json_file)
            print(f"NEGATED DATA CREATED!")
            print(f"There are {len(augmented_train) } grammatical sentences and negations in the augmented training data.")
    else:
        print("No grammatical data found. Please fix.")
        pipe = pipeline("text-classification", model="textattack/distilbert-base-uncased-CoLA")
        tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

        negations = ['neither', 'never', 'no one', 'nobody', 'none', 'nor', 'nothing', 'nowhere', "isn't", "doesn't", "hasn't", "not", "no", "can't", "won't", "shouldn't", "wouldn't", "couldn't", "don't", "didn't", "aren't", "wasn't", "weren't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "won't", "wouldn't", "shan't", "shouldn't", "mustn't", "needn't", "mightn't"]

        grammatical_train = []
        grammatical_train_negations = []

        print(f"Creating grammatical sentences for {len(train_data)} data points...")
        for datum in tqdm(train_data):
            sent1 = datum.get_sent1()
            sent2 = datum.get_sent2()
            # first check if sentences are grammatical -- using distilbert (trained on CoLA grammaticality classification)
            if pipe(sent1)[0]['label'] == 'LABEL_1' and pipe(sent1)[0]['score'] > 0.8:
                if pipe(sent2)[0]['label'] == 'LABEL_1' and pipe(sent2)[0]['score'] > 0.8:
                    # add to list grammatical_train
                    grammatical_train.append(datum)
                    # tokenize the sentences (get word tokens)
                    tokens1 = tokenizer.tokenize(sent1)
                    tokens2 = tokenizer.tokenize(sent2)
                    if any(negation in tokens1 for negation in negations) or any(negation in tokens2 for negation in negations):
                        # add datum to new list "grammatical_train_negations"
                        grammatical_train_negations.append(datum)

        print(f"There are {len(grammatical_train)} grammatical sentences in the training data.")
        print(f"There are {len(grammatical_train_negations) } grammatical sentences with negations in the training data.")
        print(f"Percentage of grammatical sentences with negations: {len(grammatical_train_negations) / len(grammatical_train) * 100}%")
        
        serialized_data = [datum.to_dict() for datum in grammatical_train]
        
        with open(grammatical_path, "w") as json_file:
            json.dump(serialized_data, json_file)

        neg_train_data = create_negated(grammatical_train)
        augmented_train = train_data + neg_train_data
        random.shuffle(augmented_train)

        serialized_data = [datum.to_dict() for datum in augmented_train]

        with open(negated_grammatical_path, "w") as json_file:
            json.dump(serialized_data, json_file)


# check number of points which contain negations

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

negations = ['neither', 'never', 'no one', 'nobody', 'none', 'nor', 'nothing', 'nowhere', "isn't", "doesn't", "hasn't", "not", "no", "can't", "won't", "shouldn't", "wouldn't", "couldn't", "don't", "didn't", "aren't", "wasn't", "weren't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "won't", "wouldn't", "shan't", "shouldn't", "mustn't", "needn't", "mightn't"]

full_dataset_negations = []
grammatical_train_negations = []
dev_matched_negations = []

print(f"checking for negation in {len(train_data)} data points...")
for datum in tqdm(train_data):
    sent1 = datum.get_sent1()
    sent2 = datum.get_sent2()

    # tokenize the sentences (get word tokens)
    tokens1 = tokenizer.tokenize(sent1)
    tokens2 = tokenizer.tokenize(sent2)
    if any(negation in tokens1 for negation in negations) or any(negation in tokens2 for negation in negations):
        # add datum to new list
        full_dataset_negations.append(datum)

print(f"checking for negation in {len(grammatical_train)} data points...")
for datum in tqdm(grammatical_train):
    sent1 = datum.get_sent1()
    sent2 = datum.get_sent2()

    # tokenize the sentences (get word tokens)
    tokens1 = tokenizer.tokenize(sent1)
    tokens2 = tokenizer.tokenize(sent2)
    if any(negation in tokens1 for negation in negations) or any(negation in tokens2 for negation in negations):
        # add datum to new list "grammatical_train_negations"
        grammatical_train_negations.append(datum)

print(f"checking for negation in {len(dev_matched)} data points...")
for datum in tqdm(dev_matched):
    sent1 = datum.get_sent1()
    sent2 = datum.get_sent2()

    # tokenize the sentences (get word tokens)
    tokens1 = tokenizer.tokenize(sent1)
    tokens2 = tokenizer.tokenize(sent2)
    if any(negation in tokens1 for negation in negations) or any(negation in tokens2 for negation in negations):
        # add datum to new list "grammatical_train_negations"
        dev_matched_negations.append(datum)

# write the dev_matched_negations to a json file called neg_only_dev_matched.json
serialized_data = [datum.to_dict() for datum in dev_matched_negations]

with open("./multinli_1.0/neg_only_dev_matched.json", "w") as json_file:
    json.dump(serialized_data, json_file)
    print(f"dev set with negations written to file.")

print(f"There are {len(full_dataset_negations) } sentences with negations in the training data.")
print(f"Percentage of grammatical sentences with negations: {len(full_dataset_negations) / len(train_data) * 100}%")

print(f"There are {len(grammatical_train_negations) } grammatical sentences with negations in the training data.")
print(f"Percentage of grammatical sentences with negations: {len(grammatical_train_negations) / len(grammatical_train) * 100}%")

print(f"There are {len(dev_matched_negations) } sentences with negations in the dev matched data.")
print(f"Percentage of grammatical sentences with negations: {len(dev_matched_negations) / len(dev_matched) * 100}%")