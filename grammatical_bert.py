# Doing the same experiment except only using grammatical sentences and their negations
# 

from background_and_models import *

grammatical_path = "./multinli_1.0/grammatical_dataset.json"
negated_grammatical_path = "./multinli_1.0/negated_grammatical_dataset.json"
matched_path = "./multinli_1.0/multinli_1.0_dev_matched.jsonl"

# load grammatical data if it exists
if os.path.exists(grammatical_path):
    with open(grammatical_path, "r") as json_file:
        loaded_data = json.load(json_file)
    grammatical_train = [Datum.from_dict(data) for data in loaded_data]

# load negated grammatical_data if it exists
if os.path.exists(negated_grammatical_path):
    with open(negated_grammatical_path, "r") as json_file:
        loaded_data = json.load(json_file)
    augmented_train = [Datum.from_dict(data) for data in loaded_data]

print(f"There are {len(grammatical_train)} grammatical sentences in the training data.")
print(f"There are {len(augmented_train) } grammatical sentences and negations in the augmented training data.")
    

# load matched dev set
dev_matched = load_data(matched_path)

f1s_matched = {}

# add f1 scores to dictionary

print("\n\n\n\n\n\n")

print("Full finetune of DistilBERT with grammatical only data")

print("\n\n\n---------------------------------\n\n\n")


classifier_base, classifier_aug, optimizer_base, optimizer_aug = initialize_DB_models() # initialize models
criterion = "cross_entropy"

print(f"\n[base] DistilBERT Fine-tuned with last layer unfrozen:\n")
train_classifier_BERT(grammatical_train, classifier_base, optimizer_base, criterion, 3)
f1s_matched["baseDistilBERT"] = test_classifier_BERT(dev_matched, classifier_base)

print(f"\n[aug] DistilBERT Fine-tuned with last layer unfrozen:\n")
train_classifier_BERT(augmented_train, classifier_aug, optimizer_aug, criterion, 3)
f1s_matched["augDistilBERT"] = test_classifier_BERT(dev_matched, classifier_aug)

torch.save(classifier_base.state_dict(), "./models/gramdistilBERT_baseCE.pth")
torch.save(classifier_aug.state_dict(), "./models/gramdistilBERT_augCE.pth")

print("\n\n\n---------------------------------\n\n\n")


for key in f1s_matched.keys():
    print(f"F1 score for {key} on matched dev set: {f1s_matched[key]}")

