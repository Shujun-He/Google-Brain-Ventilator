import numpy as np


class FeatureProcessor():
    def __init__(self,df,categorial_features):
        self.df=df
        self.categorial_features=categorial_features

        self.uniques={}
        for feature_name in self.categorial_features:
            self.uniques[feature_name]=df[feature_name].unique()


    def process(self):
        for feature_name in self.categorial_features:
            for i in range(len(self.df)):
                one_hot_features=np.zeros(len(self.uniques[feature_name]))
                index=np.where(self.uniques[feature_name]==self.df[feature_name].iloc[i])[0][0]
                one_hot_features[index]=1
                self.df[feature_name].iloc[i]=[one_hot_features]
        return self.df
