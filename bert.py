from background_and_models import *
from sklearn.metrics import confusion_matrix

train_path = "./multinli_1.0/multinli_1.0_train.jsonl"
matched_path = "./multinli_1.0/multinli_1.0_dev_matched.jsonl"
mismatched_path = "./multinli_1.0/multinli_1.0_dev_mismatched.jsonl"
augmented_path = "./multinli_1.0/augmented_dataset.json"

# No need to split data. It has been pre-split into train and dev sets (see MNLI; Williams et al., 2018)
train_data = load_data(train_path)
dev_matched = load_data(matched_path)
# dev_mismatched = load_data(mismatched_path)

# load augmented data if it exists, else create it and write to the file
if os.path.exists(augmented_path):
    with open(augmented_path, "r") as json_file:
        loaded_data = json.load(json_file)
    augmented_train = [Datum.from_dict(data) for data in loaded_data]

else:
    neg_train_data = create_negated(train_data)
    random.shuffle(neg_train_data)
    augmented_train = train_data + neg_train_data
    serialized_data = [datum.to_dict() for datum in augmented_train]
    with open(augmented_path, "w") as json_file:
        json.dump(serialized_data, json_file)


f1s_matched = {}

print("\n\n\n\n\n\n")

print("Retraining distilbert base and distilbert aug...")

print("\n\n\n---------------------------------\n\n\n")


classifier_base, classifier_aug, optimizer_base, optimizer_aug = initialize_DB_models() # initialize models
criterion = "cross_entropy"

print(f"\n[baseCE] DistilBERT Fine-tuned with last layer unfrozen:\n")
train_classifier_BERT(train_data, classifier_base, optimizer_base, criterion, 3)
f1s_matched["baseDistilBERT_CE"] = test_classifier_BERT(dev_matched, classifier_base)

print(f"\n[augCE] DistilBERT Fine-tuned with last layer unfrozen:\n")
train_classifier_BERT(augmented_train, classifier_aug, optimizer_aug, criterion, 3)
f1s_matched["augDistilBERT_CE"] = test_classifier_BERT(dev_matched, classifier_aug)

torch.save(classifier_base.state_dict(), "./models/baseDistilBERT_CE.pth")
torch.save(classifier_aug.state_dict(), "./models/augDistilBERT_CE.pth")

print("\n\n\n---------------------------------\n\n\n")


for key in f1s_matched.keys():
    print(f"F1 score for {key} on matched dev set: {f1s_matched[key]}")
    # print(f"F1 score for {key} on mismatched dev set: {f1s_mismatched[key]}")

# because it has failed to save so many times, I'm writing all the other code here too
def test_wrong_pred(test_data, classifier):
    all_outputs = []
    all_targets = []
    random.shuffle(test_data)
    with torch.no_grad():
        for datum in tqdm(test_data):
            sent1 = datum.get_sent1()
            sent2 = datum.get_sent2()
            classifier_outputs = classifier(sent1, sent2)
            classifier_outputs = classifier_outputs.cpu().numpy()
            outputs = np.argmax(classifier_outputs, axis=1)
            targets = [labels.index(datum.get_gold_label())]
            all_outputs.extend(outputs)
            all_targets.extend(targets)
    
    print(classification_report(all_targets, all_outputs, target_names=labels))
    cm = confusion_matrix(all_targets, all_outputs)
    
    # generate examples where the model predicts the wrong label and correct label
    wrong_predictions = [(test_data[i].get_sent1(), test_data[i].get_sent2(), labels[all_outputs[i]], labels[all_targets[i]]) for i in range(len(all_outputs)) if all_outputs[i] != all_targets[i]]
    correct_predictions = [(test_data[i].get_sent1(), test_data[i].get_sent2(), labels[all_outputs[i]], labels[all_targets[i]]) for i in range(len(all_outputs)) if all_outputs[i] == all_targets[i]]
    
    print(f"Wrong predictions: {len(wrong_predictions)}")
    print(f"Correct predictions: {len(correct_predictions)}")
    print(f'Confusion matrix: {cm}\n')

    # adding this to add to list
    return wrong_predictions, correct_predictions

# write the predictions to a file in a json readable format
def write_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        json.dump(predictions, f, indent=4)

# get the wrong predictions for each model
print("Base Model:")
base_wrong_preds, base_correct_preds = test_wrong_pred(dev_matched, classifier_base)
print("\nAug Model:")
aug_wrong_preds, aug_correct_preds = test_wrong_pred(dev_matched, classifier_aug)

# write all predictions to a file
write_predictions_to_file(base_wrong_preds, './outputs/base_wrong_preds.json')
write_predictions_to_file(base_correct_preds, './outputs/base_correct_preds.json')
write_predictions_to_file(aug_wrong_preds, './outputs/aug_wrong_preds.json')
write_predictions_to_file(aug_correct_preds, './outputs/aug_correct_preds.json')