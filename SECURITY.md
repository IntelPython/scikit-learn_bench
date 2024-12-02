pip install scikit-learn-intelex# Classification OOP API Example

# loading sample dataset
from pycaret.datasets import get_data
data = get_data('juice')

# init setup
from pycaret.classification import ClassificationExperiment
s = ClassificationExperiment()
s.setup(data, target = 'Purchase', session_id = 123)

# model training and selection
best = s.compare_models()

# evaluate trained model
s.evaluate_model(best)

# predict on hold-out/test set
pred_holdout = s.predict_model(best)

# predict on new data
new_data = data.copy().drop('Purchase', axis = 1)
predictions = s.predict_model(best, data = new_data)

# save model
s.save_model(best, 'best_pipeline')pycaret.datasetspycaret.classification# Classification Functional API Example

# loading sample dataset
from pycaret.datasets import get_data
data = get_data('juice')

# init setup
from pycaret.classification import *
s = setup(data, target = 'Purchase', session_id = 123)

# model training and selection
best = compare_models()

# evaluate trained model
evaluate_model(best)

# predict on hold-out/test set
pred_holdout = predict_model(best)

# predict on new data
new_data = data.copy().drop('Purchase', axis = 1)
predictions = predict_model(best, data = new_data)

# save model
save_model(best, 'best_pipeline')# default version
docker run -p 8888:8888 pycaret/slim

# full version
docker run -p 8888:8888 pycaret/full# Security Policy
Intel is committed to rapidly addressing security vulnerabilities affecting our customers and providing clear guidance on the solution, impact, severity and mitigation. 

## Reporting a Vulnerability
Please report any security vulnerabilities in this project utilizing the guidelines [here](https://www.intel.com/content/www/us/en/security-center/vulnerability-handling-guidelines.html).
