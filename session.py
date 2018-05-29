# coding: utf-8
get_ipython().magic('pylab')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import main
data = main.load_data()
vects = main.load_google_news_vectors()

w2v_v_scores, w2v_clusterings = main.get_vmeasure_curve_and_clusterings(data, main.get_w2v_tranform(vects, tfidf=False))
tfidf_v_scores, w2v_clusterings = main.get_vmeasure_curve_and_clusterings(data, main.tfidf_transform)

main.v_measure_figure()
plot(w2v_v_scores)
plot(tfidf_v_scores)