"""
train_model.py

This script:
1. Loads your training data
2. Extracts features from words
3. Trains Decision Tree and Naive Bayes classifiers
4. Evaluates and compares them
5. Saves the best trained model
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# FEATURE EXTRACTION

def extract_features(word):
    import re
    features = {}
    if not word or len(word.strip()) == 0:
        return {f'feature_{i}': 0 for i in range(58)}

    word = word.strip()
    lower = word.lower()
    vowels = 'aeiou'

    # 4 Basic Structure of word
    features['length'] = len(word)
    features['is_short'] = 1 if len(word) <= 2 else 0
    features['is_long'] = 1 if len(word) > 10 else 0
    features['starts_capital'] = 1 if word[0].isupper() else 0

    # 4 character composition 
    vowel_count = sum(1 for c in lower if c in vowels)
    consonant_count = sum(1 for c in lower if c.isalpha() and c not in vowels)
    alpha_count = sum(1 for c in word if c.isalpha())

    features['vowel_ratio'] = vowel_count / len(word) if len(word) > 0 else 0
    features['consonant_ratio'] = consonant_count / len(word) if len(word) > 0 else 0
    features['alpha_ratio'] = alpha_count / len(word) if len(word) > 0 else 0
    features['ends_vowel'] = 1 if lower and lower[-1] in vowels else 0

    # 8 filipino n grams 
    features['has_ng'] = 1 if 'ng' in lower else 0
    features['starts_ng'] = 1 if lower.startswith('ng') else 0
    features['ends_ng'] = 1 if lower.endswith('ng') else 0
    features['has_ay'] = 1 if 'ay' in lower else 0
    features['has_ka'] = 1 if 'ka' in lower else 0
    features['has_na'] = 1 if 'na' in lower else 0
    features['has_mga'] = 1 if 'mga' in lower else 0
    features['has_oo'] = 1 if 'oo' in lower else 0

    # 6 english n grams
    features['has_th'] = 1 if 'th' in lower else 0
    features['has_ch'] = 1 if 'ch' in lower else 0
    features['has_qu'] = 1 if 'qu' in lower else 0
    features['has_sh'] = 1 if 'sh' in lower else 0
    features['has_ing'] = 1 if 'ing' in lower else 0
    features['has_tion'] = 1 if 'tion' in lower else 0

    # 10 morphology
    fil_prefixes = ['nag', 'mag', 'naka', 'maka', 'pag', 'na', 'ma', 'pa', 'um', 'in']

    features['has_fil_prefix'] = 1 if any(lower.startswith(p) for p in fil_prefixes) else 0
    features['ends_in'] = 1 if lower.endswith('in') else 0
    features['ends_an'] = 1 if lower.endswith('an') else 0
    features['ends_han'] = 1 if lower.endswith('han') else 0

    eng_suffixes = ['ing', 'ed', 'er', 'ly', 'tion', 'ment', 'ness', 'able', 'ous']

    features['has_eng_suffix'] = 1 if any(lower.endswith(s) for s in eng_suffixes) else 0
    features['ends_ed'] = 1 if lower.endswith('ed') else 0
    features['ends_ly'] = 1 if lower.endswith('ly') else 0
    features['ends_tion'] = 1 if lower.endswith('tion') else 0
    features['ends_able'] = 1 if lower.endswith('able') else 0
    features['ends_ous'] = 1 if lower.endswith('ous') else 0

    # 3 phonetic sounds
    consonant_clusters = ['str', 'spr', 'scr', 'thr', 'chr', 'spl', 'shr']

    features['has_consonant_cluster'] = 1 if any(cc in lower for cc in consonant_clusters) else 0

    cv_pattern = len(re.findall(r'[bcdfghjklmnpqrstvwxyz][aeiou]', lower))

    features['cv_ratio'] = cv_pattern / len(word) if len(word) > 0 else 0
    features['has_double_vowel'] = 1 if re.search(r'[aeiou]{2,}', lower) else 0

    # 4 letter frequency
    rare_fil = 'cfjqvxz'

    features['has_rare_filipino_letter'] = 1 if any(c in lower for c in rare_fil) else 0
    features['starts_k'] = 1 if lower.startswith('k') else 0
    features['starts_c'] = 1 if lower.startswith('c') else 0
    features['starts_b'] = 1 if lower.startswith('b') else 0

    # 2 special cases
    features['has_reduplication'] = 1 if '-' in word and len(word.split('-')) == 2 else 0
    features['has_numbers'] = 1 if any(c.isdigit() for c in word) else 0

    # 4 Symbol/Punctuation detection 
    features['is_punctuation'] = 1 if all(not c.isalnum() for c in word) else 0
    features['has_special_chars'] = 1 if any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`' for c in word) else 0
    features['is_pure_number'] = 1 if word.replace('.', '').replace(',', '').replace('-', '').isdigit() else 0
    features['special_char_ratio'] = sum(1 for c in word if not c.isalnum()) / len(word) if len(word) > 0 else 0
    
    # 4 Abbreviation patterns
    features['is_all_caps'] = 1 if word.isupper() and len(word) > 1 else 0
    features['has_periods'] = 1 if '.' in word else 0
    features['is_abbreviation'] = 1 if (len(word) <= 5 and word.isupper()) or word.count('.') >= 2 else 0
    features['all_caps_short'] = 1 if word.isupper() and len(word) <= 4 else 0
    
    # 2 Named Entity hints
    features['is_title_case'] = 1 if word.istitle() and len(word) > 2 else 0
    features['all_consonants'] = 1 if consonant_count == len(word) and len(word) > 1 else 0
    
    # 1 Mixed alphanumeric 
    features['is_alphanumeric_mix'] = 1 if any(c.isdigit() for c in word) and any(c.isalpha() for c in word) else 0

    # 2 Common English / Filipino words
    common_eng = ['i', 'am', 'is', 'are', 'the', 'and', 'but', 'or', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'this']
    common_fil = ['ng', 'na', 'ka', 'mga', 'ako', 'siya', 'ito', 'iyan', 'ang', 'sa', 'ay', 'si', 'ni']
    
    features['is_common_eng_word'] = 1 if lower in common_eng else 0
    features['is_common_fil_word'] = 1 if lower in common_fil else 0
    
    # OTH Detection 
    
    # Has only non-letter characters
    features['no_letters'] = 1 if not any(c.isalpha() for c in word) else 0
    
    # Unusual character patterns (likely OTH)
    features['has_multiple_hyphens'] = 1 if word.count('-') >= 2 else 0
    features['has_underscore'] = 1 if '_' in word else 0

    expressions = ['haha', 'hehe', 'hihi', 'huhu', 'lol', 'omg', 'wtf', 
                   'grr', 'ugh', 'hmm', 'uhh', 'ahh', 'ohh', 'hahaha', 
                   'hehehe', 'zzz', 'aww', 'ugh', 'meh', 'yay', 'nope']
    
    features['is_expression'] = 1 if lower in expressions else 0

    return features


# LOAD AND PREPARE DATA

def load_and_prepare_data(csv_file):
    """Load data from CSV and prepare for training safely."""

    print("Loading validated data...")
    df = pd.read_csv(csv_file)

    df['word'] = df['word'].astype(str).replace('nan', '')
    print(f"Loaded {len(df)} words")
    print("\nTag distribution:")
    print(df['tag'].value_counts())

    # Extract features for each word
    print("\nExtracting features from words...")

    feature_list = [extract_features(word) for word in df['word']]
    labels = df['tag'].tolist()

    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(feature_list)

    # Define fixed feature order
    FEATURE_ORDER = [
    'length', 'is_short', 'is_long', 'starts_capital',
    'vowel_ratio', 'consonant_ratio', 'alpha_ratio', 'ends_vowel',
    'has_ng', 'starts_ng', 'ends_ng', 'has_ay', 'has_ka', 'has_na', 'has_mga', 'has_oo',
    'has_th', 'has_ch', 'has_qu', 'has_sh', 'has_ing', 'has_tion',
    'has_fil_prefix', 'ends_in', 'ends_an', 'ends_han',
    'has_eng_suffix', 'ends_ed', 'ends_ly', 'ends_tion', 'ends_able', 'ends_ous',
    'has_consonant_cluster', 'cv_ratio', 'has_double_vowel',
    'has_rare_filipino_letter', 'starts_k', 'starts_c', 'starts_b',
    'has_reduplication', 'has_numbers',
    'is_punctuation', 'has_special_chars', 'is_pure_number', 'special_char_ratio',
    'is_all_caps', 'has_periods', 'is_abbreviation', 'all_caps_short',
    'is_title_case', 'all_consonants',
    'is_alphanumeric_mix',
    'is_common_eng_word', 'is_common_fil_word',
    'no_letters', 'has_multiple_hyphens', 'has_underscore', 'is_expression'
    ]   

    # Keep only columns that exist in features (in case extract_features changes)
    FEATURE_ORDER = [f for f in FEATURE_ORDER if f in features_df.columns]
    features_df = features_df[FEATURE_ORDER]

    # Fill any missing values with 0
    features_df = features_df.fillna(0)

    # Convert to NumPy arrays
    X = features_df.values
    y = np.array(labels)

    print(f"\nFeature extraction complete!")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   (Rows = words, Columns = features)")

    # Save to CSV for inspection
    features_df['tag'] = y
    features_df.to_csv('data/training_features.csv', index=False)
    print("Saved feature matrix to 'data/training_features.csv'")

    return X, y


# TRAIN AND EVALUATE RANDOM FOREST

def train_random_forest(X, y):
    """Train and evaluate only a Random Forest classifier"""
    
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    # Split data= 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split complete:")
    print(f"   Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # TRAIN RANDOM FOREST
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)

    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(
        n_estimators=2000,      
        max_depth=50,        
        min_samples_split=3,   
        min_samples_leaf=1,     
        max_features='sqrt',    
        class_weight = {                             
            'ENG': 1.0,
            'FIL': 1.0,
            'OTH': 2.5                           
        },
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )

    print("\nTraining...")
    rf_model.fit(X_train, y_train)

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_pred = rf_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"\nValidation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_pred = rf_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, test_pred, zero_division=0))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred, labels=['ENG', 'FIL', 'OTH'])
    print("           Predicted")
    print("Actual     ENG  FIL  OTH")
    for i, label in enumerate(['ENG', 'FIL', 'OTH']):
        print(f"{label:6}   {cm[i][0]:4} {cm[i][1]:4} {cm[i][2]:4}")
    
    return rf_model, 'Random Forest', test_acc


# SAVE MODEL

def save_model(model, model_name, test_accuracy):
    """Save the trained model to disk"""
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("\nCreated 'models' directory")
    
    # Save model
    model_path = 'models/pinoybot_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {model_path}")
    
    # Save model info
    info_path = 'models/model_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"\nFeatures Used:\n")
        f.write(f"- Word length\n")
        f.write(f"- Vowel/consonant ratios\n")
        f.write(f"- Has ng/th/ay/ka/qu/ph/ch patterns\n")
        f.write(f"- Starts with c, ch (common in English)\n")
        f.write(f"- Filipino prefixes (nag-, mag-, naka-, etc.)\n")
        f.write(f"- English suffixes (-ing, -ed, -tion, etc.)\n")
        f.write(f"- Morphological patterns (e.g. reduplication)\n")
        f.write(f"- Capitalization and symbol features\n")
        f.write(f"- Numeric and punctuation indicators\n")
        f.write(f"- Short word indicator\n")
    
    print(f"Model info saved to: {info_path}")


# MAIN EXECUTION

if __name__ == "__main__":
    print("="*60)
    print("PINOYBOT MODEL TRAINING")
    print("="*60)
    print()
    
    # Load data
    X, y = load_and_prepare_data('data/prepared_data.csv')
    
    # Train models
    best_model, model_name, test_accuracy = train_random_forest(X, y)
    
    # Save best model
    save_model(best_model, model_name, test_accuracy)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nYour model is ready to use!")
