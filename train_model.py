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

# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_features(word):
    """
    Extract features from a single word.
    These features help the model identify the language.
    """
    features = {}
    
    # Handle empty strings
    if len(word) == 0:
        return {f'feature_{i}': 0 for i in range(20)}
    
    vowels = 'aeiouAEIOU'
    
    # Feature 1: Word length
    features['length'] = len(word)
    
    # Feature 2: Vowel ratio
    vowel_count = sum(1 for c in word if c in vowels)
    features['vowel_ratio'] = vowel_count / len(word)
    
    # Feature 3: Has 'ng' (common in Filipino)
    features['has_ng'] = 1 if 'ng' in word.lower() else 0
    
    # Feature 4: Has 'th' (common in English)
    features['has_th'] = 1 if 'th' in word.lower() else 0
    
    # Feature 5: Starts with capital letter
    features['starts_capital'] = 1 if word[0].isupper() else 0
    
    # Feature 6: Ends with vowel
    features['ends_vowel'] = 1 if word[-1].lower() in vowels else 0
    
    # Feature 7: All uppercase
    features['all_upper'] = 1 if word.isupper() else 0
    
    # Feature 8: Has numbers
    features['has_numbers'] = 1 if any(c.isdigit() for c in word) else 0
    
    # Feature 9: All punctuation/symbols
    features['all_symbols'] = 1 if all(not c.isalnum() for c in word) else 0
    
    # Feature 10: Ratio of consonants
    consonant_count = sum(1 for c in word if c.isalpha() and c not in vowels)
    features['consonant_ratio'] = consonant_count / len(word)
    
    # Feature 11: Has common English suffixes
    eng_suffixes = ['ing', 'ed', 'tion', 'ly', 'ness', 'ment']
    features['has_eng_suffix'] = 1 if any(word.lower().endswith(s) for s in eng_suffixes) else 0
    
    # Feature 12: Has common Filipino prefixes
    fil_prefixes = ['nag', 'mag', 'naka', 'maka', 'pag', 'na']
    features['has_fil_prefix'] = 1 if any(word.lower().startswith(p) for p in fil_prefixes) else 0
    
    # Feature 13: Has reduplication pattern (common in Filipino)
    features['has_reduplication'] = 1 if '-' in word and len(word.split('-')) == 2 else 0
    
    # Feature 14: Has 'ay' (very common in Filipino)
    features['has_ay'] = 1 if 'ay' in word.lower() else 0
    
    # Feature 15: Has 'qu' (more common in English)
    features['has_qu'] = 1 if 'qu' in word.lower() else 0
    
    # Feature 16: Has 'ka' (common in Filipino)
    features['has_ka'] = 1 if 'ka' in word.lower() else 0
    
    # Feature 17: Ends with 'in' (common Filipino verb ending)
    features['ends_in'] = 1 if word.lower().endswith('in') else 0
    
    # Feature 18: Ends with 'an' (common Filipino noun ending)
    features['ends_an'] = 1 if word.lower().endswith('an') else 0
    
    # Feature 19: Very short word (1-2 chars)
    features['very_short'] = 1 if len(word) <= 2 else 0
    
    # Feature 20: Has 'ph' (common in English)
    features['has_ph'] = 1 if 'ph' in word.lower() else 0
    
    return features

# ============================================
# LOAD AND PREPARE DATA
# ============================================

def load_and_prepare_data(csv_file):
    """Load data from CSV and prepare for training"""
    
    print("Loading training data...")
    df = pd.read_csv(csv_file)
    
    print(f"✅ Loaded {len(df)} words")
    print(f"\nTag distribution:")
    print(df['tag'].value_counts())
    
    # Extract features for each word
    print("\nExtracting features from words...")
    feature_list = []
    labels = []
    
    for idx, row in df.iterrows():
        word = row['word']
        label = row['tag']
        
        features = extract_features(word)
        feature_list.append(list(features.values()))
        labels.append(label)
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} words...")
    
    X = np.array(feature_list)
    y = np.array(labels)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   (Rows = words, Columns = features)")
    
    return X, y

# ============================================
# TRAIN AND EVALUATE MODELS
# ============================================

def train_models(X, y):
    """Train both Decision Tree and Naive Bayes, compare them"""
    
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    # Split data: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Split temp into validation (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"\n✅ Data split complete:")
    print(f"   Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    models = {}
    
    # ===== TRAIN DECISION TREE =====
    print("\n" + "="*60)
    print("TRAINING DECISION TREE")
    print("="*60)
    
    dt_model = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    print("\nTraining...")
    dt_model.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    dt_val_pred = dt_model.predict(X_val)
    dt_val_acc = accuracy_score(y_val, dt_val_pred)
    
    print(f"\n✅ Validation Accuracy: {dt_val_acc:.4f} ({dt_val_acc*100:.2f}%)")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, dt_val_pred, zero_division=0))
    
    models['Decision Tree'] = {
        'model': dt_model,
        'val_accuracy': dt_val_acc,
        'val_predictions': dt_val_pred
    }
    
    # ===== TRAIN NAIVE BAYES =====
    print("\n" + "="*60)
    print("TRAINING NAIVE BAYES")
    print("="*60)
    
    nb_model = GaussianNB()
    
    print("\nTraining...")
    nb_model.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    nb_val_pred = nb_model.predict(X_val)
    nb_val_acc = accuracy_score(y_val, nb_val_pred)
    
    print(f"\n✅ Validation Accuracy: {nb_val_acc:.4f} ({nb_val_acc*100:.2f}%)")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, nb_val_pred, zero_division=0))
    
    models['Naive Bayes'] = {
        'model': nb_model,
        'val_accuracy': nb_val_acc,
        'val_predictions': nb_val_pred
    }
    
    # ===== COMPARE AND SELECT BEST MODEL =====
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    print("\nValidation Accuracy Comparison:")
    for name, result in models.items():
        print(f"  {name:20}: {result['val_accuracy']:.4f} ({result['val_accuracy']*100:.2f}%)")
    
    best_model_name = max(models, key=lambda x: models[x]['val_accuracy'])
    best_model = models[best_model_name]['model']
    
    print(f"\n🏆 WINNER: {best_model_name}")
    
    # ===== FINAL TEST EVALUATION =====
    print("\n" + "="*60)
    print(f"FINAL TEST EVALUATION - {best_model_name}")
    print("="*60)
    
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, test_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred, labels=['ENG', 'FIL', 'OTH'])
    print("           Predicted")
    print("Actual     ENG  FIL  OTH")
    for i, label in enumerate(['ENG', 'FIL', 'OTH']):
        print(f"{label:6}   {cm[i][0]:4} {cm[i][1]:4} {cm[i][2]:4}")
    
    return best_model, best_model_name, test_acc

# ============================================
# SAVE MODEL
# ============================================

def save_model(model, model_name, test_accuracy):
    """Save the trained model to disk"""
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("\n✅ Created 'models' directory")
    
    # Save model
    model_path = 'models/pinoybot_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Model saved to: {model_path}")
    
    # Save model info
    info_path = 'models/model_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"PinoyBot Language Identifier\n")
        f.write(f"="*40 + "\n\n")
        f.write(f"Model Type: {model_name}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Number of Features: 20\n")
        f.write(f"\nFeatures Used:\n")
        f.write(f"- Word length\n")
        f.write(f"- Vowel/consonant ratios\n")
        f.write(f"- Character n-grams (ng, th, ay, ka, qu, ph)\n")
        f.write(f"- Filipino prefixes (nag-, mag-, etc.)\n")
        f.write(f"- English suffixes (-ing, -ed, -tion, etc.)\n")
        f.write(f"- Morphological patterns\n")
        f.write(f"- Capitalization features\n")
    
    print(f"✅ Model info saved to: {info_path}")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("PINOYBOT MODEL TRAINING")
    print("="*60)
    print()
    
    # Load data
    X, y = load_and_prepare_data('data/training_data.csv')
    
    # Train models
    best_model, model_name, test_accuracy = train_models(X, y)
    
    # Save best model
    save_model(best_model, model_name, test_accuracy)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE! 🎉")
    print("="*60)
    print("\nYour model is ready to use!")
    print("\nNext steps:")
    print("1. Update pinoybot.py with the extract_features() function")
    print("2. Test with: python pinoybot.py")
    print("3. If accuracy is low, try adding more features")
    print()