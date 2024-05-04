import polars as pl
from sklearn.tree import DecisionTreeRegressor

df = pl.read_csv('data/train.csv')
df_test = pl.read_csv('data/test.csv')
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

def process(df):
    return df.with_columns(
        pl.col('Age').fill_null(pl.col('Age').mean())
    ).with_columns(
        pl.col('Sex').map_elements(lambda sex: 1 if sex == 'male' else 0, return_dtype=pl.UInt32)
    )

df = process(df)
df_test = process(df_test)

x = df[features]
y = df['Survived']
input = df_test[features]

model = DecisionTreeRegressor(max_leaf_nodes=22, random_state=0)

model.fit(x, y)

output = pl.Series(model.predict(input)).map_elements(lambda x: 1 if x >= 0.5 else 0, return_dtype=pl.UInt32)

submission = pl.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': output})

submission.write_csv('submission.csv')
