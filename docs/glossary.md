## Bio
**One-hot AA**  
Vector with 20 slots (one per amino acid). Put a 1 in the slot for the amino acid present and 0 everywhere else — one such vector for the wild-type letter, one for the mutant letter (or their combination, depending on how it's set up).

**FoldX ΔΔG**  
Estimate of how much a mutation destabilizes a protein's 3D structure. ΔΔG is the change in folding stability the mutation causes, measured in energy units (kcal/mol). A positive number means the mutant is less stable (more likely to unfold) than the wild-type; a negative number means it's more stable.
- If the FoldX number alone predicts mechanism almost as well as ESM-2's embeddings do, that tells you ESM-2 isn't adding much beyond "how destabilizing is this mutation"; the classifier could just be relying on stability, not real mechanism understanding. 
- If ESM-2's embeddings do noticeably better than the FoldX number alone, that's evidence ESM-2 is picking up on something beyond simple structural stability.

## Stats  
  
**F1 score**  
F1 score(0.0 to 1.0) measures how well a binary classifier is working. It balances precision and recall.

**Macro-F1**  
F1 for multiple classes. Treats every class equally (without respect to size), hence is more fair.

**Spearman rho**   
Measures how well two things agree on ranking, not on exact values. If you rank all the mutations by their predicted stability score, and separately rank them by their actual lab-measured stability, Spearman rho tells you how similar those two rankings are — 1 means perfect agreement, 0 means no relationship, -1 means perfectly backwards. It's used here instead of a plain correlation because it doesn't require the predicted and actual values to be on the same scale, just to move in the same order.

**Leakage diagnostic**  
A test that checks whether a model’s apparent performance is real signal or just data leakage (the model cheating by seeing information it shouldn’t have at test time).

**Cross-validation**  
Test how well a model will work on new data it hasn’t seen before.  
You have a dataset. Instead of training the model on all of it and then hoping it works on future data, you:

- Split the data into several pieces (called folds).
- Train the model on most of the pieces.
- Test it on the piece you left out.
- Repeat this so that every piece gets to be the “test” piece once.
- Average the results.
This gives a more honest estimate of how good the model actually is.

**Gene-split cross-validation**  
The train/test split is done at the gene level, not at the individual variant level.  How it works
- Group all variants by the gene they come from.
- Split the genes into folds (usually 5-fold GroupKFold).
- Every variant belonging to a gene that is assigned to the test fold goes into the test set.
No variant from a test gene ever appears in the training set.

**Family-split cross-validation**
Even if you keep whole genes out of the test set (gene-split), genes that belong to the same protein family are still very similar to each other. So a model can still “cheat” by recognising the family instead of learning a real general rule about mutations.

- Genes are grouped by their protein family (using Pfam).
- Entire families are held out together.
- When a family is in the test set, the model has never seen any gene from that family during training.

The model must make predictions about completely new families of proteins it has never encountered before.

**Label-permutation tests**  
Randomly shuffle the true labels many times and re-run the model. This shows how often you would get the observed performance just by chance.

**Naive baseline**
Answers: Is this model better than doing nothing?  

**WT-only baseline**  
Answers: Is this model better than just recognising which gene/family it is?

### The core experiment  

**Gene-split − family-split gap**  
Measure performance twice - once under family-split, and once under gene-split. The gap is simply:
gene-split score − family-split score  
A large positive gap means performance dropped a lot when families were held out, and that most of the original signal was leakage.

**Bootstrap**  
Statistical technique for estimating how reliable a number is when you only have one dataset. Core idea  - 
Instead of collecting new data over and over, you pretend your existing data is the whole population and create many 
new “fake” datasets by randomly sampling from it with replacement. Some examples get picked multiple times. Some examples get left out.

You then recalculate your statistic (in this case the performance gap) on each of these new datasets.  
After doing this hundreds or thousands of times, you have a whole distribution of possible values for your statistic. From that distribution you can see:

- How much the number tends to jump around
- A confidence interval (e.g. “we’re 95% confident the true gap is between X and Y”)

**Paired bootstrap**  
Paired means both scores come from the exact same variants. The only thing that changed is how the data was split.
Because the two results are linked (paired), you can directly compare them instead of treating them 
as two independent experiments. 

**Paired bootstrap on the gap**  
This is the actual leakage diagnostic. It measures the difference in performance between gene-split and family-split, 
and checks whether that difference is statistically reliable. A large, significant drop when moving to family-split is evidence of leakage.

**WT-only baseline**  
A control that predicts the gene’s dominant disease mechanism class (GOF / DN / LOF) using only the wild-type ESM-2 embedding of the protein — no mutant sequence and no delta (mutant − wild-type).  
It answers the question:
“How much mechanism signal can be read just from the identity / family of the gene, without any mutation information?”

**Does the strong-(ish?) performance of the WT-only baseline under gene-split CV survive when we hold out entire Pfam families?**  
The WT-only baseline predicts a mutation’s disease mechanism using only the embedding of the unmutated (wild-type) protein. The question is whether that 
performance reflects genuine biological signal or is largely an artifact of homology. To test this, the same model is re-evaluated under 
family-split cross-validation. The performance dropped.
