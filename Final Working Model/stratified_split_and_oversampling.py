import pandas as pd
import numpy as np
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, accuracy_score, multilabel_confusion_matrix
from skmultilearn.problem_transform import ClassifierChain, LabelPowerset
from skmultilearn.ensemble import RakelD
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset_end_ver.csv', index_col=[0])

# Define feature and label columns
x_columns = ['n_r_ecg_p_08', 'inf_im', 'ZSN_A', 'GB', 'SEX', 'K_BLOOD', 'NA_BLOOD', 'zab_leg_01', 'nr_03', 'LID_S_n',
             'ROE', 'AST_BLOOD', 'ritm_ecg_p_04', 'B_BLOK_S_n', 'O_L_POST', 'DLIT_AG', 'zab_leg_02', 'GIPER_NA', 'INF_ANAM', 'ant_im',
             'endocr_01', 'IM_PG_P', 'S_AD_ORIT', 'fibr_ter_08', 'FK_STENOK', 'STENOK_AN', 'AGE', 'ritm_ecg_p_02', 'lat_im',
             'fibr_ter_01', 'IBS_POST', 'L_BLOOD', 'TIME_B_S', 'FIB_G_POST', 'post_im', 'fibr_ter_07', 'n_p_ecg_p_11',
             'ALT_BLOOD', 'ritm_ecg_p_06', 'NITR_S', 'fibr_ter_06', 'D_AD_ORIT', 'fibr_ter_03', 'nr_11', 'GIPO_K']

y_columns = ['FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM',
             'P_IM_STEN', 'alive', 'cardiogenic shock', 'pulmonary edema', 'myocardial rupture', 'progress of congestive heart failure',
             'thromboembolism', 'asystole', 'ventricular fibrillation']

# Features and labels
X = df[x_columns]
y = df[y_columns]

# Create stratified splits
def create_split(nfolds=5, order=2):
    k_fold = IterativeStratification(n_splits=nfolds, order=order)
    splits = list(k_fold.split(X, y))
    fold_splits = np.zeros(df.shape[0], dtype=int)
    for i in range(nfolds):
        fold_splits[splits[i][1]] = i
    df['Split'] = fold_splits
    df_folds = []
    for fold in range(nfolds):
        df_fold = df.copy()
        train_df = df_fold[df_fold.Split != fold].drop('Split', axis=1).reset_index(drop=True)
        test_df = df_fold[df_fold.Split == fold].drop('Split', axis=1).reset_index(drop=True)
        df_folds.append((train_df, test_df))
    return df_folds

df_folds = create_split(nfolds=5, order=2)

classifiers = {
    'Binary Relevance': BinaryRelevance(classifier=RandomForestClassifier(), require_dense=[False, True]),
    'Classifier Chain': ClassifierChain(classifier=RandomForestClassifier(), require_dense=[False, True]),
    'Label Powerset': LabelPowerset(classifier=RandomForestClassifier(), require_dense=[False, True]),
    'RAKEL-D': RakelD(base_classifier=RandomForestClassifier())
}

# Initialize a dictionary to store results
results = {clf_name: {'accuracy': [], 'f1_score': [], 'hamming_loss': [], 'jaccard_samples': [], 'jaccard_micro': []}
           for clf_name in classifiers.keys()}

for clf_name, classifier in classifiers.items():
    print(f"Evaluating {clf_name}:")
    for i, (train_df, test_df) in enumerate(df_folds):
        full_counts = {}
        for lbl in y_columns:
            count = train_df[lbl].sum()
            full_counts[lbl] = count

        label_counts = list(zip(full_counts.keys(), full_counts.values()))
        label_counts = np.array(sorted(label_counts, key=lambda x: -x[1]))
        label_counts = pd.DataFrame(label_counts, columns=['label', 'full_count'])
        label_counts.set_index('label', inplace=True)
        label_counts['full_count'] = pd.to_numeric(label_counts['full_count'])
        total = label_counts['full_count'].sum()
        avg = total / len(y_columns)

        def find_sample_ratio(x):
            x = int(x)
            if x >= avg:
                return 1
            else:
                return int(np.round(avg / x))

        label_counts['oversampling_ratio'] = label_counts['full_count'].apply(find_sample_ratio)

        def get_sample_ratio(row):
            ratio = 1
            for l in y_columns:
                r = label_counts.oversampling_ratio.loc[l]
                if r > ratio:
                    ratio = r
            return ratio

        rows = train_df.values.tolist()
        print("Starting rows:", len(rows))
        oversampled_rows = [row for row in rows for _ in range(get_sample_ratio(row))]
        print("Oversampled total:", len(oversampled_rows))

        train_df = pd.DataFrame(oversampled_rows, columns=train_df.columns)

        X_train = train_df[x_columns]
        y_train = train_df[y_columns]
        X_test = test_df[x_columns]
        y_test = test_df[y_columns]

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        classifier.fit(X_train, y_train)

        y_hat = classifier.predict(X_test).toarray()
        y_test = y_test.to_numpy()

        # Calculate overall performance metrics
        br_f1 = f1_score(y_test, y_hat, average='micro')
        br_hamm = hamming_loss(y_test, y_hat)
        br_jaccard_samples = jaccard_score(y_test, y_hat, average='samples')
        br_jaccard_micro = jaccard_score(y_test, y_hat, average='micro')
        br_test_accuracy = accuracy_score(y_test, y_hat)

        # Store results
        results[clf_name]['accuracy'].append(br_test_accuracy)
        results[clf_name]['f1_score'].append(br_f1)
        results[clf_name]['hamming_loss'].append(br_hamm)
        results[clf_name]['jaccard_samples'].append(br_jaccard_samples)
        results[clf_name]['jaccard_micro'].append(br_jaccard_micro)

        print(f'Split number: {i}')
        print(f'{clf_name} Testing accuracy: {round(br_test_accuracy, 3)}')
        print(f'{clf_name} F1-score: {round(br_f1, 3)}')
        print(f'{clf_name} Hamming Loss: {round(br_hamm, 3)}')
        print(f'{clf_name} Jaccard score (samples): {round(br_jaccard_samples, 3)}')
        print(f'{clf_name} Jaccard score (micro): {round(br_jaccard_micro, 3)}')

        # Calculate confusion matrices for each label
        conf_matrices = multilabel_confusion_matrix(y_test, y_hat)
        for idx, cm in enumerate(conf_matrices):
            print(f"Confusion Matrix for label {y_columns[idx]}:\n{cm}\n")

    print('\n')

# Print results dictionary to verify its content
print("Results dictionary content:")
print(results)

# Plotting the results using bar plots
metrics_to_plot = ['accuracy', 'f1_score', 'hamming_loss', 'jaccard_samples', 'jaccard_micro']

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    for clf_name in classifiers.keys():
        if len(results[clf_name][metric]) > 0:
            plt.bar([f"Fold {i+1}" for i in range(len(df_folds))], results[clf_name][metric], label=clf_name)
        else:
            print(f"No data for {clf_name} and metric {metric}")
    plt.title(f'Comparison of {metric} across different classifiers')
    plt.xlabel('Fold')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting confusion matrices for the first split
for clf_name, classifier in classifiers.items():
    train_df, test_df = df_folds[0]
    X_train = train_df[x_columns]
    y_train = train_df[y_columns]
    X_test = test_df[x_columns]
    y_test = test_df[y_columns]

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    classifier.fit(X_train, y_train)

    y_hat = classifier.predict(X_test).toarray()
    y_test = y_test.to_numpy()

    conf_matrices = multilabel_confusion_matrix(y_test, y_hat)
