import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--train_output", type=str)
parser.add_argument("--test_output", type=str)
args = parser.parse_args()

df = pd.read_csv(args.input_data)

# 1. Supprimer l'identifiant
df = df.drop(columns=["Student_ID"])

# 2. Supprimer les lignes sans cible
df = df.dropna(subset=["Burnout_Risk_Level"])

# 3. Encodage ORDINAL - Prompt_Engineering_Skill (Beginner/Intermediate/Advanced)
skill_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
df["Prompt_Engineering_Skill"] = df["Prompt_Engineering_Skill"].map(skill_map)

# 4. Encodage ORDINAL - Year_of_Study (Freshman/Sophomore/Junior/Senior)
year_map = {"Freshman": 0, "Sophomore": 1, "Junior": 2, "Senior": 3}
df["Year_of_Study"] = df["Year_of_Study"].map(year_map)

# 5. Encodage BINAIRE - Paid_Subscription (True/False)
df["Paid_Subscription"] = df["Paid_Subscription"].astype(int)

# 6. Cible - Burnout_Risk_Level (Low/Medium/High)
risk_map = {"Low": 0, "Medium": 1, "High": 2}
df["Burnout_Risk_Level"] = df["Burnout_Risk_Level"].map(risk_map)

# 7. Perceived_AI_Dependency et Anxiety_Level_During_Exams sont DÉJÀ numériques -> rien à faire

# 8. Encodage NOMINAL pour le reste (Major_Category, Primary_Use_Case, Institutional_Policy)
remaining_categorical = df.select_dtypes(include="object").columns.tolist()
for col in remaining_categorical:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 9. Valeurs manquantes restantes
df = df.fillna(df.median(numeric_only=True))

# 10. Vérification : aucune colonne ne doit avoir de NaN après mapping
assert df.isnull().sum().sum() == 0, "Il reste des valeurs manquantes après encodage !"

# 11. Split stratifié
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["Burnout_Risk_Level"]
)

train.to_csv(f"{args.train_output}/train.csv", index=False)
test.to_csv(f"{args.test_output}/test.csv", index=False)

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Distribution Burnout (train):\n{train['Burnout_Risk_Level'].value_counts()}")