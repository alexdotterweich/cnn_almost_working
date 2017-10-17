Using <b> distinct categories</b>, the model achieves ~96% validation and training accuracy. 

Using <b> non-distinct categories</b>, the training accuracy is stuck at ~58% validation accuracy 
while the training accuract gradually increases to ~80% after 22 epochs.

Some of the non-distinct categories are not syntatically similar. For example, the model lumps various
airline names with names of popular hotels and classifies them both under the label "travel." A potential
solution to increase the validation accuracy might be to create more labels and merge them at the end. 
