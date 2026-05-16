import sys
sys.path.insert(0, '.')
from model.recommender import load_and_prepare_data, recommend_papers

df, t, v, c = load_and_prepare_data()
results = recommend_papers('Machine Learning', df, t, v, c, top_n=10)
for r in results:
    print(f"{r['rank']}. [{r['year']}] Rel:{r['relevance_score']}% Sim:{r['similarity_score']}% Rec:{r['recency_score']}% | {r['title'][:60]}")
