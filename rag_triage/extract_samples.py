# """Extract sample cases from SYNAPSE for testing."""
# import pandas as pd
# df = pd.read_csv('../data/SYNAPSE_An Expert Annotated Dataset of Patient symptoms and Demographics.csv')
# otc = df[df['Final Recommendation'] == 'OTC Drug'].sample(5, random_state=42)
# doc = df[df['Final Recommendation'] == 'Doctor Consultation'].sample(5, random_state=42)
# print('=== OTC Drug (5 samples) ===')
# for _, row in otc.iterrows():
#     print(row['Symptoms'], '|', row['Gender'], '|', row['Age'], '|', row['Duration'], '|', row['Severity'], '->', row['Final Recommendation'])
# print()
# print('=== Doctor Consultation (5 samples) ===')
# for _, row in doc.iterrows():
#     print(row['Symptoms'], '|', row['Gender'], '|', row['Age'], '|', row['Duration'], '|', row['Severity'], '->', row['Final Recommendation'])
