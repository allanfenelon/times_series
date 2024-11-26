# Content available in page https://analyticsindiamag.com/a-guide-to-inferencing-with-bayesian-network-in-python/
# PACKAGE INSTALLATION AND LOADING
# pip install pgmpy
# from pgmpy.model2s import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
# import pylab as plt
import matplotlib.pyplot as plt
# Defining Bayesian Structure
# arcs from the cause to the effect
model = BayesianNetwork([('hasKnowledgment', 'correctAnswer_Q01')])
# Defining the CPDs:
# PRIORS
cpd_hasKnowledgment = TabularCPD(variable='hasKnowledgment',
                                 variable_card=2, values=[[.5], [.5]], state_names={"hasKnowledgment": ["YES", "NO"]})  # 0- yes, 1-no
print(cpd_hasKnowledgment)
# LIKELIHOODS
cpd_correctAnswer_Q01 = TabularCPD(variable='correctAnswer_Q01',
                                   variable_card=2, values=[[.9, .25],  # 0- yes, 1-no
                                                            [.1, .75]],
                                   evidence=['hasKnowledgment'], evidence_card=[2],
                                   state_names={"hasKnowledgment": ["YES", "NO"],
                                                "correctAnswer_Q01": ["YES", "NO"]})
print(cpd_correctAnswer_Q01)
# Associating the CPDs with the network structure.
model.add_cpds(cpd_hasKnowledgment, cpd_correctAnswer_Q01)
model.check_model()

# PLOT THE MODEL
nx.draw(model, with_labels=True)
plt.show()
# plt.savefig('BBN_model.png')
# plt.close()

# BBN (for qualitative analysis)
# COMPUTING THE POSTERIOR/MARGINAL DISTRIBUTIONS

infer = VariableElimination(model)
posterior_p = infer.query(['hasKnowledgment'], evidence={
                          'correctAnswer_Q01': 'YES'})
print(posterior_p)

# INVOLVING A SECOND QUESTION, MORE DIFFICULT TO ANSWER...
model2 = BayesianNetwork([('hasKnowledgment', 'correctAnswer_Q01'),
                          ('hasKnowledgment', 'correctAnswer_Q02')])  # arcs from the cause to the effect
# Defining the CPDs:
# PRIORS
cpd_hasKnowledgment = TabularCPD(variable='hasKnowledgment',
                                 variable_card=2, values=[[.5], [.5]], state_names={"hasKnowledgment": ["YES", "NO"]})  # 0- yes, 1-no
print(cpd_hasKnowledgment)
# LIKELIHOODS
cpd_correctAnswer_Q01 = TabularCPD(variable='correctAnswer_Q01',
                                   variable_card=2, values=[[.9, .25],  # 0- yes, 1-no
                                                            [.1, .75]],
                                   evidence=['hasKnowledgment'], evidence_card=[2],
                                   state_names={"hasKnowledgment": ["YES", "NO"],
                                                "correctAnswer_Q01": ["YES", "NO"]})
print(cpd_correctAnswer_Q01)
cpd_correctAnswer_Q02 = TabularCPD(variable='correctAnswer_Q02',
                                   variable_card=2, values=[[.9, .25],  # 0- yes, 1-no
                                                            [.1, .75]],
                                   evidence=['hasKnowledgment'], evidence_card=[2],
                                   state_names={"hasKnowledgment": ["YES", "NO"],
                                                "correctAnswer_Q02": ["YES", "NO"]})
print(cpd_correctAnswer_Q02)
# Associating the CPDs with the network structure.
model2.add_cpds(cpd_hasKnowledgment, cpd_correctAnswer_Q01,
                cpd_correctAnswer_Q02)
model2.check_model()
# BBN (for qualitative analysis)
# COMPUTING THE POSTERIOR/MARGINAL DISTRIBUTIONS
# from pgmpy.inference import VariableElimination

infer = VariableElimination(model2)
posterior_p2 = infer.query(['hasKnowledgment'], evidence={
                           'correctAnswer_Q01': 'YES', 'correctAnswer_Q02': 'YES'})
print(posterior_p2)
