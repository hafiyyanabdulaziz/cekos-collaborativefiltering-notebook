import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

property_interaction_df = pd.DataFrame(pd.read_csv('dataset_interaction_excel.csv',index_col=0))
interactions_matrix = pd.pivot_table(property_interaction_df, values='ratings', index='user_id', columns='property_id', aggfunc=np.max).fillna(0)

def similar_users(user_id, interactions_matrix):
    similarity = []
    for user in range(0, interactions_matrix.shape[0]):
        sim = cosine_similarity([interactions_matrix.loc[user_id]], [interactions_matrix.loc[interactions_matrix.index[user]]])
        similarity.append((interactions_matrix.index[user], sim))
    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity]
    similarity_score = [tup[1] for tup in similarity]
    most_similar_users.remove(user_id)
    similarity_score.remove(similarity_score[0])
    return most_similar_users, similarity_score

def recommendations(user_id, num_of_property, user_item_interactions):
    most_similar_users = similar_users(user_id, user_item_interactions)[0]
    property_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[user_id] > 0)]))
    recommendations = []
    already_interacted = property_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_property:
            similar_user_property_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)]))
            recommendations.extend(list(similar_user_property_ids.difference(already_interacted)))
            already_interacted = already_interacted.union(similar_user_property_ids)
        else:
            break
    return recommendations[:num_of_property]

print(recommendations('5bfaed099a291d00012035eb', 5, interactions_matrix))