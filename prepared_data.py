"""
prepare_data.py

Converts validated annotations to 3-class training format (ENG, FIL, OTH)
"""

import pandas as pd

def convert_to_training_label(label, special_tags, corrected_label, corrected_special_tags):
    """
    Convert detailed labels to ENG/FIL/OTH based on project specs
    
    Rules from specs:
    - FIL → FIL (including code-switched words)
    - ENG (without special tags) → ENG
    - Everything else -> OTH (SYM, NUM, UNK, ENG with NE/ABB, etc.)
    """
    
    # Use corrected values if they exist, otherwise use original
    final_label = corrected_label if pd.notna(corrected_label) else label
    final_special = corrected_special_tags if pd.notna(corrected_special_tags) else special_tags
    
    # FIL stays FIL (includes code-switched per specs)
    if final_label == 'FIL':
        return 'FIL'
    
    # ENG without special tags stays ENG
    if final_label == 'ENG' and (pd.isna(final_special) or final_special == ''):
        return 'ENG'
    
    # Everything else is OTH
    return 'OTH'

# ============================================
# MAIN EXECUTION
# ============================================

print("="*60)
print("PREPARING TRAINING DATA")
print("="*60)

# Load validated data
print("\n1. Loading validated data...")

#validated_data_group30.csv
df = pd.read_csv('Data/full_rawdata.csv')


print(f"Loaded {len(df)} words from {df['sentence_id'].nunique()} sentences")



# Show original tag distribution
print("\n2. Original label distribution:")
print(df['label'].value_counts())

# Apply conversion
print("\n3. Converting to 3-class format (ENG, FIL, OTH)...")
df['training_label'] = df.apply(
    lambda row: convert_to_training_label(
        row['label'], 
        row['special_tags'],
        row.get('corrected_label'),
        row.get('corrected_special_tags')
    ),
    axis=1
)

# Create training dataset
training_data = df[['word', 'training_label']].copy()
training_data.columns = ['word', 'tag']

# Save
output_file = 'data/prepared_data.csv'
training_data.to_csv(output_file, index=False)

print(f"Saved to: {output_file}")

# Show statistics
print("\n4. Training data statistics:")
print(f"Total words: {len(training_data)}")
print(f"\nTag distribution:")
print(training_data['tag'].value_counts())

# Show some examples
print("\n5. Example conversions (first 20 words):")
print("-" * 60)
comparison = df[['word', 'label', 'special_tags', 'training_label']].head(20)
print(comparison.to_string(index=False))

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE!")
print("="*60)